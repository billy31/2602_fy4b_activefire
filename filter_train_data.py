#!/usr/bin/env python3
"""
filter_train_data.py

该脚本用于：
1. 解压training文件夹下的压缩包（支持多层嵌套压缩包）
2. 根据landsat-filter结果筛选符合if_within_geo条件的训练数据
3. 同步代码到GitHub

压缩包解压规则：
- .zip -> 解压到 raw/ 文件夹
- _mask.zip -> 解压到 mask/ 文件夹
- _masks_derivates.zip -> 解压到 mask_label/ 文件夹
- 解压后删除原压缩包
"""

import os
import re
import sys
import glob
import zipfile
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set


# ============== 配置 ==============
TRAINING_DIR = "/root/autodl-tmp/training"
METADATA_DIR = "/root/autodl-tmp/metadata"
LANDSAT_FILTER_CSV = os.path.join(METADATA_DIR, "landsat-filter.csv")

# GitHub同步脚本路径
SYNC_SCRIPT = "/root/codes/fire0226/selfCodes/sync_to_github.sh"
WORKSPACE_DIR = "/root/codes/fire0226/selfCodes"


# ============== 函数1：解压压缩包 ==============

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    解压zip文件到指定目录
    
    Args:
        zip_path: zip文件路径
        extract_to: 解压目标目录
        
    Returns:
        是否成功解压
    """
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"    解压失败 {os.path.basename(zip_path)}: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    删除文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否成功删除
    """
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"    删除失败 {os.path.basename(file_path)}: {e}")
        return False


def process_nested_zips(folder_path: str) -> None:
    """
    递归处理文件夹下的所有压缩包
    - .zip 解压到 raw/ 文件夹
    - _mask.zip 解压到 mask/ 文件夹
    - _masks_derivates.zip 解压到 mask_label/ 文件夹
    
    Args:
        folder_path: 要处理的文件夹路径
    """
    # 查找所有zip文件
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return
    
    print(f"  发现 {len(zip_files)} 个压缩包")
    
    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path)
        
        # 判断压缩包类型并确定解压目标
        if zip_name.endswith("_masks_derivates.zip"):
            extract_to = os.path.join(folder_path, "mask_label")
            zip_type = "mask_label"
        elif zip_name.endswith("_mask.zip"):
            extract_to = os.path.join(folder_path, "mask")
            zip_type = "mask"
        else:
            # 普通 .zip 解压到 raw
            extract_to = os.path.join(folder_path, "raw")
            zip_type = "raw"
        
        print(f"    解压 [{zip_type}] {zip_name} -> {extract_to}")
        
        # 解压
        if extract_zip(zip_path, extract_to):
            # 解压成功后删除原文件
            if delete_file(zip_path):
                print(f"      -> 已解压并删除")
            
            # 检查解压后的文件夹内是否还有压缩包（递归处理）
            process_nested_zips(extract_to)


def extract_region_archives(training_dir: str, specific_regions: Optional[List[str]] = None) -> None:
    """
    函数1：查找并解压地区压缩包
    
    Args:
        training_dir: training文件夹路径
        specific_regions: 指定要处理的地区列表，None表示处理全部
    """
    print("\n【步骤1】查找并解压压缩包...")
    
    # 获取要处理的地区列表
    if specific_regions:
        regions = specific_regions
    else:
        # 获取所有子文件夹和zip文件
        items = os.listdir(training_dir)
        regions = []
        for item in items:
            item_path = os.path.join(training_dir, item)
            # 如果是文件夹，直接加入
            if os.path.isdir(item_path):
                regions.append(item)
            # 如果是地区压缩包（如 Asia1.zip），解压后处理
            elif item.endswith('.zip') and not item.endswith(('_mask.zip', '_masks_derivates.zip')):
                region_name = item[:-4]  # 去掉.zip
                extract_to = os.path.join(training_dir, region_name)
                print(f"  发现地区压缩包: {item} -> 解压到 {region_name}/")
                if extract_zip(item_path, extract_to):
                    delete_file(item_path)
                    regions.append(region_name)
    
    # 处理每个地区的内部压缩包
    for region in regions:
        region_path = os.path.join(training_dir, region)
        if not os.path.exists(region_path):
            print(f"  跳过不存在的地区: {region}")
            continue
        
        print(f"\n  处理地区: {region}")
        process_nested_zips(region_path)


# ============== 函数2：根据landsat-filter筛选数据 ==============

def extract_landsat_id_from_filename(filename: str) -> Optional[str]:
    """
    从tif文件名中提取Landsat图像ID
    
    例如：
    - LC08_L1GT_117027_20200809_20200809_01_RT_p00153.tif
    -> LC08_L1GT_117027_20200809_20200809_01_RT
    
    Args:
        filename: tif文件名
        
    Returns:
        Landsat图像ID或None
    """
    # 匹配 Landsat ID 格式: LC08_L1GT_117027_20200809_20200809_01_RT
    pattern = r'(LC\d+_L\d[A-Z]+_\d+_\d+_\d+_\d+_[A-Z]+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def load_landsat_filter(csv_path: str) -> Dict[str, bool]:
    """
    加载landsat-filter.csv，构建landsat_id到if_within_geo的映射
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        字典: {landsat_id: if_within_geo}
    """
    df = pd.read_csv(csv_path)
    
    # 从filename列提取landsat_id（去掉_MTL.txt后缀）
    landsat_map = {}
    for _, row in df.iterrows():
        mtl_filename = row['filename']
        # 去掉 _MTL.txt 后缀
        landsat_id = mtl_filename.replace('_MTL.txt', '')
        landsat_map[landsat_id] = row['if_within_geo']
    
    return landsat_map


def filter_tif_files_by_landsat(
    training_dir: str, 
    landsat_map: Dict[str, bool],
    specific_regions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    函数2：根据landsat-filter筛选tif文件
    
    Args:
        training_dir: training文件夹路径
        landsat_map: landsat_id到if_within_geo的映射
        specific_regions: 指定要处理的地区列表，None表示处理全部
        
    Returns:
        包含所有tif文件筛选结果的DataFrame
    """
    print("\n【步骤2】根据landsat-filter筛选tif文件...")
    
    # 获取要处理的地区列表
    if specific_regions:
        regions = specific_regions
    else:
        regions = [d for d in os.listdir(training_dir) 
                  if os.path.isdir(os.path.join(training_dir, d))]
    
    records = []
    
    for region in regions:
        region_path = os.path.join(training_dir, region)
        if not os.path.exists(region_path):
            continue
        
        print(f"\n  处理地区: {region}")
        
        # 查找所有tif文件
        tif_files = []
        for subdir in ['raw', 'mask', 'mask_label']:
            subdir_path = os.path.join(region_path, subdir)
            if os.path.exists(subdir_path):
                tif_files.extend(glob.glob(os.path.join(subdir_path, "*.tif")))
        
        print(f"    找到 {len(tif_files)} 个tif文件")
        
        # 统计信息
        stats = {"total": 0, "within_geo": 0, "not_in_geo": 0, "unknown": 0}
        
        for tif_path in tif_files:
            filename = os.path.basename(tif_path)
            subdir = os.path.basename(os.path.dirname(tif_path))
            
            # 提取Landsat ID
            landsat_id = extract_landsat_id_from_filename(filename)
            
            if landsat_id is None:
                if_within_geo = None
                stats["unknown"] += 1
            elif landsat_id in landsat_map:
                if_within_geo = landsat_map[landsat_id]
                if if_within_geo:
                    stats["within_geo"] += 1
                else:
                    stats["not_in_geo"] += 1
            else:
                # landsat_id不在filter中，标记为unknown
                if_within_geo = None
                stats["unknown"] += 1
            
            stats["total"] += 1
            
            record = {
                "region": region,
                "subdir": subdir,
                "filename": filename,
                "landsat_id": landsat_id,
                "tif_path": tif_path,
                "if_within_geo": if_within_geo
            }
            records.append(record)
        
        print(f"    统计: 总计={stats['total']}, 在范围内={stats['within_geo']}, "
              f"不在范围内={stats['not_in_geo']}, 未知={stats['unknown']}")
    
    df = pd.DataFrame(records)
    return df


# ============== GitHub同步 ==============

def sync_to_github() -> bool:
    """
    同步代码到GitHub
    
    Returns:
        是否同步成功
    """
    print("\n【步骤3】同步到GitHub...")
    
    try:
        os.chdir(WORKSPACE_DIR)
        
        # 检查是否有变更
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("  没有变更需要同步")
            return True
        
        # 添加所有变更
        subprocess.run(["git", "add", "-A"], check=True)
        
        # 提交
        commit_msg = f"Update filter_train_data - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # 推送
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ 同步成功！")
            return True
        else:
            print(f"  ❌ 推送失败: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git操作失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 同步失败: {e}")
        return False


# ============== 主函数 ==============

def main(specific_regions: Optional[List[str]] = None):
    """
    主函数：运行函数1+2，并同步到GitHub
    
    Args:
        specific_regions: 指定要处理的地区列表，None表示处理全部
    """
    print("=" * 70)
    print("训练数据筛选工具 - 根据Landsat-filter筛选tif文件")
    print("=" * 70)
    
    # 步骤1：解压压缩包
    extract_region_archives(TRAINING_DIR, specific_regions)
    
    # 步骤2：加载landsat-filter并筛选tif文件
    if not os.path.exists(LANDSAT_FILTER_CSV):
        print(f"错误：找不到landsat-filter文件: {LANDSAT_FILTER_CSV}")
        return
    
    print(f"\n加载landsat-filter: {LANDSAT_FILTER_CSV}")
    landsat_map = load_landsat_filter(LANDSAT_FILTER_CSV)
    print(f"  共加载 {len(landsat_map)} 条Landsat记录")
    
    df = filter_tif_files_by_landsat(TRAINING_DIR, landsat_map, specific_regions)
    
    if df.empty:
        print("\n警告：没有找到任何tif文件")
        return
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("筛选结果统计")
    print("=" * 70)
    print(f"总tif文件数: {len(df)}")
    print(f"在if_within_geo范围内的: {df['if_within_geo'].sum()}")
    print(f"不在范围内的: {(df['if_within_geo'] == False).sum()}")
    print(f"未知(未匹配到landsat-filter): {df['if_within_geo'].isna().sum()}")
    
    print("\n各地区分布：")
    region_stats = df.groupby("region")["if_within_geo"].agg(['count', 'sum', lambda x: (x == False).sum(), lambda x: x.isna().sum()])
    region_stats.columns = ['total', 'within_geo', 'not_in_geo', 'unknown']
    print(region_stats)
    
    # 步骤3：同步到GitHub
    sync_to_github()
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练数据筛选工具")
    parser.add_argument(
        "--regions", 
        nargs="+", 
        help="指定要处理的地区（如 Asia1 Asia2），不指定则处理全部"
    )
    
    args = parser.parse_args()
    
    # 运行主函数
    main(specific_regions=args.regions)
