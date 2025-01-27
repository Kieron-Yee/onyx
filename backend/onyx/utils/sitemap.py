"""
网站地图处理工具模块
本模块提供了一系列用于处理和解析网站地图(sitemap)的工具函数，
可以从网站的robots.txt和sitemap.xml中提取URL列表。
"""

import re
import xml.etree.ElementTree as ET
from typing import Set
from urllib.parse import urljoin

import requests

from onyx.utils.logger import setup_logger

logger = setup_logger()


def _get_sitemap_locations_from_robots(base_url: str) -> Set[str]:
    """Extract sitemap URLs from robots.txt
    从robots.txt中提取网站地图URL
    
    参数:
        base_url: str - 网站的基础URL
        
    返回:
        Set[str] - 包含网站地图URL的集合
    """
    sitemap_urls: set = set()
    try:
        robots_url = urljoin(base_url, "/robots.txt")
        resp = requests.get(robots_url, timeout=10)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    sitemap_urls.add(sitemap_url)
    except Exception as e:
        logger.warning(f"Error fetching robots.txt: {e}")
    return sitemap_urls


def _extract_urls_from_sitemap(sitemap_url: str) -> Set[str]:
    """Extract URLs from a sitemap XML file
    从网站地图XML文件中提取URL
    
    参数:
        sitemap_url: str - 网站地图的URL地址
        
    返回:
        Set[str] - 从网站地图中提取的URL集合
    """
    urls: set[str] = set()
    try:
        resp = requests.get(sitemap_url, timeout=10)
        if resp.status_code != 200:
            return urls

        root = ET.fromstring(resp.content)

        # Handle both regular sitemaps and sitemap indexes
        # Remove namespace for easier parsing
        # 处理常规网站地图和网站地图索引
        # 移除命名空间以便于解析
        namespace = re.match(r"\{.*\}", root.tag)
        ns = namespace.group(0) if namespace else ""

        if root.tag == f"{ns}sitemapindex":
            # This is a sitemap index
            # 这是一个网站地图索引
            for sitemap in root.findall(f".//{ns}loc"):
                if sitemap.text:
                    sub_urls = _extract_urls_from_sitemap(sitemap.text)
                    urls.update(sub_urls)
        else:
            # This is a regular sitemap
            # 这是一个常规网站地图
            for url in root.findall(f".//{ns}loc"):
                if url.text:
                    urls.add(url.text)

    except Exception as e:
        logger.warning(f"Error processing sitemap {sitemap_url}: {e}")

    return urls


def list_pages_for_site(site: str) -> list[str]:
    """Get list of pages from a site's sitemaps
    获取网站地图中所有页面的列表
    
    参数:
        site: str - 要处理的网站URL
        
    返回:
        list[str] - 包含所有找到的页面URL的列表
    """
    site = site.rstrip("/")
    all_urls = set()

    # Try both common sitemap locations
    # 尝试常见的网站地图位置
    sitemap_paths = ["/sitemap.xml", "/sitemap_index.xml"]
    for path in sitemap_paths:
        sitemap_url = urljoin(site, path)
        all_urls.update(_extract_urls_from_sitemap(sitemap_url))

    # Check robots.txt for additional sitemaps
    # 检查robots.txt中是否有其他网站地图
    sitemap_locations = _get_sitemap_locations_from_robots(site)
    for sitemap_url in sitemap_locations:
        all_urls.update(_extract_urls_from_sitemap(sitemap_url))

    return list(all_urls)
