#!/usr/bin/env python3
"""
filter_landsat_fy_geo.py

该脚本用于：
1. 解压metadata地区文件夹下的压缩包（如果有）
2. 读取MTL元数据文件，提取图像覆盖范围信息
3. 根据FY-4B卫星覆盖范围筛选在范围内的Landsat影像
4. 输出结果到CSV和Excel文件
"""

import os
import re
import glob
import zipfile
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# ============== 配置 ==============
METADATA_DIR = "/root/autodl-tmp/metadata"
OUTPUT_CSV = os.path.join(METADATA_DIR, "landsat-filter.csv")
OUTPUT_EXCEL = os.path.join(METADATA_DIR, "landsat-filter.xlsx")

# FY-4B卫星覆盖范围（地球同步轨道，以104.7°E为中心）
# 覆盖范围：经度约45°E - 165°E，纬度约60°S - 60°N
FY4B_COVERAGE = {
    "min_lon": 45.0,
    "max_lon": 165.0,
    "min_lat": -60.0,
    "max_lat": 60.0,
}

# 地区文件夹列表（排除zipfile和输出文件）
REGION_FOLDERS = [
    "Africa", "Asia1", "Asia2", "Asia3", "Asia4", "Asia5",
    "Europe", "North_America1", "North_America2", "Oceania", "South_America"
]


# ============== 函数1：解压压缩包并读取MTL文件 ==============

def find_and_extract_zips(region_dir: str) -> None:
    """
    检查指定地区文件夹下是否有压缩包，有则解压到当前文件夹
    
    Args:
        region_dir: 地区文件夹路径
    """
    # 支持的压缩格式
    zip_patterns = ["*.zip", "*.tar.gz", "*.tgz", "*.gz"]
    
    for pattern in zip_patterns:
        zip_files = glob.glob(os.path.join(region_dir, pattern))
        for zip_path in zip_files:
            try:
                print(f"  发现压缩包: {os.path.basename(zip_path)}")
                if zip_path.endswith('.zip'):
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(region_dir)
                    print(f"    -> 已解压ZIP: {os.path.basename(zip_path)}")
                # 如果需要支持tar.gz等格式，可以在这里扩展
            except Exception as e:
                print(f"    -> 解压失败: {e}")


def parse_mtl_file(mtl_path: str) -> Optional[Dict]:
    """
    解析MTL文件，提取图像覆盖范围信息
    
    Args:
        mtl_path: MTL文件路径
        
    Returns:
        包含图像范围信息的字典，解析失败返回None
    """
    try:
        with open(mtl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取四个角的经纬度
        def extract_float(pattern: str, text: str) -> Optional[float]:
            match = re.search(pattern + r'\s*=\s*([-\d.]+)', text)
            return float(match.group(1)) if match else None
        
        corner_ul_lat = extract_float('CORNER_UL_LAT_PRODUCT', content)
        corner_ul_lon = extract_float('CORNER_UL_LON_PRODUCT', content)
        corner_ur_lat = extract_float('CORNER_UR_LAT_PRODUCT', content)
        corner_ur_lon = extract_float('CORNER_UR_LON_PRODUCT', content)
        corner_ll_lat = extract_float('CORNER_LL_LAT_PRODUCT', content)
        corner_ll_lon = extract_float('CORNER_LL_LON_PRODUCT', content)
        corner_lr_lat = extract_float('CORNER_LR_LAT_PRODUCT', content)
        corner_lr_lon = extract_float('CORNER_LR_LON_PRODUCT', content)
        
        # 计算整体范围
        lats = [lat for lat in [corner_ul_lat, corner_ur_lat, corner_ll_lat, corner_lr_lat] if lat is not None]
        lons = [lon for lon in [corner_ul_lon, corner_ur_lon, corner_ll_lon, corner_lr_lon] if lon is not None]
        
        if not lats or not lons:
            return None
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        return {
            "min_latitude": min_lat,
            "max_latitude": max_lat,
            "min_longitude": min_lon,
            "max_longitude": max_lon,
            "center_latitude": center_lat,
            "center_longitude": center_lon,
            "corner_ul_lat": corner_ul_lat,
            "corner_ul_lon": corner_ul_lon,
            "corner_ur_lat": corner_ur_lat,
            "corner_ur_lon": corner_ur_lon,
            "corner_ll_lat": corner_ll_lat,
            "corner_ll_lon": corner_ll_lon,
            "corner_lr_lat": corner_lr_lat,
            "corner_lr_lon": corner_lr_lon,
        }
    except Exception as e:
        print(f"    解析MTL文件失败 {mtl_path}: {e}")
        return None


def extract_landsat_info(metadata_dir: str = METADATA_DIR) -> pd.DataFrame:
    """
    函数1：解压压缩包并读取所有MTL文件，提取信息到DataFrame
    
    Args:
        metadata_dir: metadata根目录
        
    Returns:
        包含所有Landsat影像信息的DataFrame
    """
    records = []
    
    for region in REGION_FOLDERS:
        region_path = os.path.join(metadata_dir, region)
        if not os.path.exists(region_path):
            print(f"跳过不存在的文件夹: {region}")
            continue
        
        print(f"\n处理地区: {region}")
        
        # 步骤1：查找并解压压缩包
        find_and_extract_zips(region_path)
        
        # 步骤2：读取所有MTL文件
        mtl_files = glob.glob(os.path.join(region_path, "*_MTL.txt"))
        print(f"  找到 {len(mtl_files)} 个MTL文件")
        
        for mtl_path in mtl_files:
            filename = os.path.basename(mtl_path)
            # 提取文件名前缀（_MTL.txt之前的部分）
            filename_prefix = filename.replace("_MTL.txt", "")
            
            # 解析MTL文件
            geo_info = parse_mtl_file(mtl_path)
            
            if geo_info:
                record = {
                    "folder": region,
                    "filename": filename,
                    "filename_prefix": filename_prefix,
                    **geo_info
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    print(f"\n总共读取了 {len(df)} 条记录")
    return df


# ============== 函数2：判断是否在FY-4B范围内 ==============

def check_in_fy4b_coverage(
    min_lat: float, 
    max_lat: float, 
    min_lon: float, 
    max_lon: float,
    coverage: Dict = FY4B_COVERAGE
) -> bool:
    """
    判断给定的图像范围是否与FY-4B覆盖范围有交集
    
    Args:
        min_lat, max_lat: 图像纬度范围
        min_lon, max_lon: 图像经度范围
        coverage: FY-4B覆盖范围字典
        
    Returns:
        是否在FY-4B范围内（有交集即算在范围内）
    """
    # 判断两个矩形是否有交集
    lat_overlap = not (max_lat < coverage["min_lat"] or min_lat > coverage["max_lat"])
    lon_overlap = not (max_lon < coverage["min_lon"] or min_lon > coverage["max_lon"])
    
    return lat_overlap and lon_overlap


def filter_by_fy4b_coverage(df: pd.DataFrame, coverage: Dict = FY4B_COVERAGE) -> pd.DataFrame:
    """
    函数2：根据FY-4B覆盖范围筛选DataFrame
    
    Args:
        df: 包含影像范围信息的DataFrame
        coverage: FY-4B覆盖范围
        
    Returns:
        添加if_within_fy4b列后的DataFrame
    """
    print(f"\n根据FY-4B覆盖范围筛选...")
    print(f"FY-4B覆盖范围: 经度 {coverage['min_lon']}°E ~ {coverage['max_lon']}°E, "
          f"纬度 {coverage['min_lat']}° ~ {coverage['max_lat']}°")
    
    df["if_within_fy4b"] = df.apply(
        lambda row: check_in_fy4b_coverage(
            row["min_latitude"], 
            row["max_latitude"],
            row["min_longitude"], 
            row["max_longitude"],
            coverage
        ),
        axis=1
    )
    
    within_count = df["if_within_fy4b"].sum()
    print(f"在FY-4B范围内的影像: {within_count} / {len(df)}")
    
    return df


# ============== 主函数 ==============

def main():
    """
    主函数：运行函数1+2，输出结果到metadata文件夹
    """
    print("=" * 60)
    print("Landsat影像范围提取与FY-4B范围筛选工具")
    print("=" * 60)
    
    # 步骤1：读取所有MTL文件信息
    print("\n【步骤1】读取MTL文件并提取地理范围信息...")
    df = extract_landsat_info(METADATA_DIR)
    
    if df.empty:
        print("错误：未找到任何有效的MTL文件")
        return
    
    # 步骤2：根据FY-4B范围筛选
    print("\n【步骤2】根据FY-4B覆盖范围筛选...")
    df = filter_by_fy4b_coverage(df)
    
    # 步骤3：输出结果
    print("\n【步骤3】保存结果...")
    
    # 选择输出列（保持与原有landsat-filter.csv兼容的格式）
    output_columns = [
        "folder", "filename", 
        "min_latitude", "max_latitude", 
        "min_longitude", "max_longitude",
        "center_latitude", "center_longitude",
        "if_within_fy4b"
    ]
    
    # 如果有额外的角点信息列，也可以选择保留
    corner_columns = [
        "corner_ul_lat", "corner_ul_lon",
        "corner_ur_lat", "corner_ur_lon",
        "corner_ll_lat", "corner_ll_lon",
        "corner_lr_lat", "corner_lr_lon"
    ]
    
    # 只保留存在的列
    final_columns = [col for col in output_columns if col in df.columns]
    df_output = df[final_columns]
    
    # 保存为CSV
    df_output.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV已保存: {OUTPUT_CSV}")
    
    # 保存为Excel
    df_output.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
    print(f"  Excel已保存: {OUTPUT_EXCEL}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("处理完成！统计信息：")
    print("=" * 60)
    print(f"总记录数: {len(df_output)}")
    print(f"在FY-4B范围内: {df_output['if_within_fy4b'].sum()}")
    print(f"不在FY-4B范围内: {(~df_output['if_within_fy4b']).sum()}")
    print("\n各地区分布：")
    print(df_output.groupby("folder")["if_within_fy4b"].agg(['count', 'sum']))


if __name__ == "__main__":
    main()
