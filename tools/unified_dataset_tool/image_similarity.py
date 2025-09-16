#!/usr/bin/env python3
"""
image similarity detection module for duplicate image removal

this module provides functionality to:
- calculate image similarity using multiple methods
- detect duplicate images based on similarity thresholds
- handle batch processing of image directories
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import threading

@dataclass
class ImageInfo:
    """container for image information and metadata"""
    path: Path
    size: int  # file size in bytes
    dimensions: Tuple[int, int]  # width, height
    hash_phash: str = None
    hash_dhash: str = None
    hash_ahash: str = None
    hash_whash: str = None
    file_hash: str = None

@dataclass
class DuplicateGroup:
    """container for a group of duplicate images"""
    primary_image: ImageInfo
    duplicates: List[ImageInfo]
    similarity_scores: Dict[str, float]  # path -> similarity score
    total_size_saved: int

class ImageSimilarityDetector:
    """main class for image similarity detection and duplicate removal"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    def __init__(self, similarity_threshold: float = 0.9, num_workers: int = None):
        """
        initialize the similarity detector
        
        args:
            similarity_threshold: threshold for considering images as duplicates (0.0-1.0)
            num_workers: number of worker threads (default: CPU count)
        """
        self.similarity_threshold = similarity_threshold
        self.progress_callback = None
        self.num_workers = num_workers or min(8, (multiprocessing.cpu_count() or 1) + 4)
        self._lock = threading.Lock()
        
    def set_progress_callback(self, callback):
        """set callback function for progress updates"""
        self.progress_callback = callback
        
    def _update_progress(self, message: str):
        """update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(message)
            
    def _get_file_hash(self, file_path: Path) -> str:
        """calculate md5 hash of file content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
            
    def _calculate_image_hashes(self, image_path: Path) -> Dict[str, str]:
        """calculate various perceptual hashes for an image"""
        try:
            with Image.open(image_path) as img:
                # convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # calculate different hash types
                hashes = {
                    'phash': str(imagehash.phash(img)),
                    'dhash': str(imagehash.dhash(img)),
                    'ahash': str(imagehash.average_hash(img)),
                    'whash': str(imagehash.whash(img))
                }
                return hashes
        except Exception as e:
            print(f"warning: failed to calculate hashes for {image_path}: {e}")
            return {}
            
    def _calculate_similarity(self, hash1: str, hash2: str) -> float:
        """calculate similarity between two perceptual hashes"""
        if not hash1 or not hash2:
            return 0.0
            
        try:
            # convert hex strings to imagehash objects
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            # calculate hamming distance
            distance = h1 - h2
            
            # convert to similarity score (0-1, where 1 is identical)
            max_distance = len(hash1) * 4  # 4 bits per hex character
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, similarity)
        except Exception:
            return 0.0
            
    def _process_single_image(self, image_path: Path) -> Optional[ImageInfo]:
        """process a single image file and extract metadata"""
        try:
            # get file size
            file_size = image_path.stat().st_size
            
            # get image dimensions
            with Image.open(image_path) as img:
                dimensions = img.size
            
            # calculate hashes
            hashes = self._calculate_image_hashes(image_path)
            
            # calculate file hash
            file_hash = self._get_file_hash(image_path)
            
            return ImageInfo(
                path=image_path,
                size=file_size,
                dimensions=dimensions,
                hash_phash=hashes.get('phash'),
                hash_dhash=hashes.get('dhash'),
                hash_ahash=hashes.get('ahash'),
                hash_whash=hashes.get('whash'),
                file_hash=file_hash
            )
            
        except Exception as e:
            print(f"warning: failed to process {image_path}: {e}")
            return None
    
    def scan_directory(self, directory: Path) -> List[ImageInfo]:
        """scan directory for images and collect metadata using multithreading"""
        # collect all image files (non-recursive to avoid subdirectory issues)
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
            
        # remove duplicates (in case of case-insensitive filesystem)
        image_files = list(set(image_files))
        total_files = len(image_files)
        self._update_progress(f"found {total_files} image files to analyze")
        
        if not image_files:
            return []
            
        images = []
        processed_count = 0
        
        # use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # submit all tasks
            future_to_path = {executor.submit(self._process_single_image, path): path 
                             for path in image_files}
            
            # collect results as they complete
            for future in as_completed(future_to_path):
                processed_count += 1
                
                if processed_count % 50 == 0 or processed_count == total_files:
                    self._update_progress(f"analyzing images: {processed_count}/{total_files}")
                
                try:
                    result = future.result()
                    if result:
                        images.append(result)
                except Exception as e:
                    path = future_to_path[future]
                    print(f"warning: failed to process {path}: {e}")
                    
        self._update_progress(f"analyzed {len(images)} images successfully")
        return images
        
    def _build_hash_index(self, images: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
        """build index of images by file hash for fast exact duplicate detection"""
        hash_index = defaultdict(list)
        for img in images:
            if img.file_hash:
                hash_index[img.file_hash].append(img)
        return hash_index
    
    def _quick_similarity_check(self, img1: ImageInfo, img2: ImageInfo) -> float:
        """fast similarity check using only one hash type (phash preferred)"""
        if img1.hash_phash and img2.hash_phash:
            return self._calculate_similarity(img1.hash_phash, img2.hash_phash)
        elif img1.hash_dhash and img2.hash_dhash:
            return self._calculate_similarity(img1.hash_dhash, img2.hash_dhash)
        elif img1.hash_ahash and img2.hash_ahash:
            return self._calculate_similarity(img1.hash_ahash, img2.hash_ahash)
        elif img1.hash_whash and img2.hash_whash:
            return self._calculate_similarity(img1.hash_whash, img2.hash_whash)
        return 0.0
    
    def _detailed_similarity_check(self, img1: ImageInfo, img2: ImageInfo) -> float:
        """detailed similarity check using all available hash types"""
        similarities = []
        
        if img1.hash_phash and img2.hash_phash:
            similarities.append(self._calculate_similarity(img1.hash_phash, img2.hash_phash))
        if img1.hash_dhash and img2.hash_dhash:
            similarities.append(self._calculate_similarity(img1.hash_dhash, img2.hash_dhash))
        if img1.hash_ahash and img2.hash_ahash:
            similarities.append(self._calculate_similarity(img1.hash_ahash, img2.hash_ahash))
        if img1.hash_whash and img2.hash_whash:
            similarities.append(self._calculate_similarity(img1.hash_whash, img2.hash_whash))
        
        return max(similarities) if similarities else 0.0
    
    def find_duplicates(self, images: List[ImageInfo]) -> List[DuplicateGroup]:
        """find duplicate image groups based on similarity using optimized algorithm"""
        if len(images) < 2:
            return []
            
        duplicate_groups = []
        processed = set()
        
        self._update_progress(f"preprocessing {len(images)} images for duplicate detection")
        
        # step 1: find exact file duplicates first (very fast)
        hash_index = self._build_hash_index(images)
        exact_duplicates_found = 0
        
        for file_hash, imgs in hash_index.items():
            if len(imgs) > 1:
                # exact duplicates found
                primary = imgs[0]
                duplicates = imgs[1:]
                
                similarity_scores = {str(dup.path): 1.0 for dup in duplicates}
                total_size_saved = sum(dup.size for dup in duplicates)
                
                group = DuplicateGroup(
                    primary_image=primary,
                    duplicates=duplicates,
                    similarity_scores=similarity_scores,
                    total_size_saved=total_size_saved
                )
                duplicate_groups.append(group)
                
                # mark all as processed
                for img in imgs:
                    processed.add(str(img.path))
                
                exact_duplicates_found += len(duplicates)
        
        if exact_duplicates_found > 0:
            self._update_progress(f"found {exact_duplicates_found} exact duplicates")
        
        # step 2: find perceptual duplicates among remaining images
        remaining_images = [img for img in images if str(img.path) not in processed]
        
        if len(remaining_images) < 2:
            self._update_progress(f"found {len(duplicate_groups)} duplicate groups total")
            return duplicate_groups
        
        total_comparisons = len(remaining_images) * (len(remaining_images) - 1) // 2
        self._update_progress(f"checking {len(remaining_images)} images for perceptual duplicates ({total_comparisons} comparisons)")
        
        comparison_count = 0
        quick_threshold = max(0.85, self.similarity_threshold - 0.05)  # lower threshold for quick check
        
        for i, img1 in enumerate(remaining_images):
            if str(img1.path) in processed:
                continue
            
            duplicates = []
            similarity_scores = {}
            
            for j in range(i + 1, len(remaining_images)):
                img2 = remaining_images[j]
                if str(img2.path) in processed:
                    continue
                
                comparison_count += 1
                if comparison_count % 5000 == 0:
                    progress_pct = (comparison_count / total_comparisons) * 100
                    self._update_progress(f"comparing images: {comparison_count}/{total_comparisons} ({progress_pct:.1f}%)")
                
                # quick similarity check first
                quick_similarity = self._quick_similarity_check(img1, img2)
                
                if quick_similarity >= quick_threshold:
                    # passed quick check, do detailed check
                    detailed_similarity = self._detailed_similarity_check(img1, img2)
                    
                    if detailed_similarity >= self.similarity_threshold:
                        duplicates.append(img2)
                        similarity_scores[str(img2.path)] = detailed_similarity
                        processed.add(str(img2.path))
            
            if duplicates:
                # calculate total size that would be saved
                total_size_saved = sum(dup.size for dup in duplicates)
                
                group = DuplicateGroup(
                    primary_image=img1,
                    duplicates=duplicates,
                    similarity_scores=similarity_scores,
                    total_size_saved=total_size_saved
                )
                duplicate_groups.append(group)
                processed.add(str(img1.path))
        
        self._update_progress(f"found {len(duplicate_groups)} duplicate groups total")
        return duplicate_groups
        
    def remove_duplicates(self, duplicate_groups: List[DuplicateGroup], 
                         keep_strategy: str = 'largest') -> Dict[str, any]:
        """
        remove duplicate images based on specified strategy
        
        args:
            duplicate_groups: list of duplicate groups
            keep_strategy: 'largest', 'smallest', 'first', 'highest_quality'
            
        returns:
            summary statistics
        """
        removed_files = []
        total_space_saved = 0
        errors = []
        
        for group in duplicate_groups:
            try:
                # determine which image to keep based on strategy
                if keep_strategy == 'largest':
                    # keep the image with largest file size
                    keep_image = max([group.primary_image] + group.duplicates, 
                                   key=lambda x: x.size)
                elif keep_strategy == 'smallest':
                    # keep the image with smallest file size
                    keep_image = min([group.primary_image] + group.duplicates, 
                                   key=lambda x: x.size)
                elif keep_strategy == 'highest_quality':
                    # keep the image with highest resolution
                    keep_image = max([group.primary_image] + group.duplicates, 
                                   key=lambda x: x.dimensions[0] * x.dimensions[1])
                else:  # 'first' or default
                    keep_image = group.primary_image
                    
                # remove all other images in the group
                to_remove = [group.primary_image] + group.duplicates
                to_remove = [img for img in to_remove if img != keep_image]
                
                for img in to_remove:
                    try:
                        # verify file exists before attempting removal
                        if not img.path.exists():
                            errors.append(f"file not found for removal: {img.path}")
                            continue
                            
                        img.path.unlink()
                        removed_files.append(str(img.path))
                        total_space_saved += img.size
                        self._update_progress(f"removed: {img.path.name}")
                    except Exception as e:
                        errors.append(f"failed to remove {img.path}: {e}")
                        
            except Exception as e:
                errors.append(f"error processing duplicate group: {e}")
                
        return {
            'removed_files': removed_files,
            'total_space_saved': total_space_saved,
            'errors': errors,
            'groups_processed': len(duplicate_groups)
        }
        
    def generate_report(self, duplicate_groups: List[DuplicateGroup]) -> str:
        """generate a detailed report of found duplicates"""
        if not duplicate_groups:
            return "no duplicate images found."
            
        report_lines = [
            f"duplicate image analysis report",
            f"{'=' * 40}",
            f"",
            f"summary:",
            f"- found {len(duplicate_groups)} duplicate groups",
            f"- total duplicate files: {sum(len(group.duplicates) for group in duplicate_groups)}",
            f"- potential space savings: {self._format_size(sum(group.total_size_saved for group in duplicate_groups))}",
            f"",
            f"duplicate groups:"
        ]
        
        for i, group in enumerate(duplicate_groups, 1):
            report_lines.append(f"")
            report_lines.append(f"group {i}:")
            report_lines.append(f"  primary: {group.primary_image.path.name} ({self._format_size(group.primary_image.size)})")
            
            for dup in group.duplicates:
                similarity = group.similarity_scores.get(str(dup.path), 0.0)
                report_lines.append(f"  duplicate: {dup.path.name} ({self._format_size(dup.size)}) - similarity: {similarity:.3f}")
                
        return "\n".join(report_lines)
        
    def _format_size(self, size_bytes: int) -> str:
        """format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

class DuplicateImageRemover:
    """high-level interface for duplicate image removal operations"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.detector = ImageSimilarityDetector(similarity_threshold)
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """set callback function for progress updates"""
        self.progress_callback = callback
        self.detector.set_progress_callback(callback)
        
    def find_and_remove_duplicates(self, directory: str, keep_strategy: str = 'largest',
                                 dry_run: bool = False) -> Dict[str, any]:
        """
        find and remove duplicate images in directory
        
        args:
            directory: path to directory to process
            keep_strategy: which duplicate to keep ('largest', 'smallest', 'first', 'highest_quality')
            dry_run: if true, only analyze without removing files
            
        returns:
            detailed results dictionary
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists() or not directory_path.is_dir():
                raise ValueError(f"invalid directory: {directory}")
                
            # scan for images
            if self.progress_callback:
                self.progress_callback("scanning directory for images...")
            images = self.detector.scan_directory(directory_path)
            
            if not images:
                return {
                    'success': True,
                    'message': 'no images found in directory',
                    'images_scanned': 0,
                    'duplicate_groups': 0,
                    'removed_files': [],
                    'total_space_saved': 0,
                    'errors': []
                }
                
            # find duplicates
            if self.progress_callback:
                self.progress_callback("analyzing images for duplicates...")
            duplicate_groups = self.detector.find_duplicates(images)
            
            # generate report
            report = self.detector.generate_report(duplicate_groups)
            
            # remove duplicates if not dry run
            removal_results = {
                'removed_files': [],
                'total_space_saved': 0,
                'errors': []
            }
            
            if not dry_run and duplicate_groups:
                if self.progress_callback:
                    self.progress_callback("removing duplicate files...")
                removal_results = self.detector.remove_duplicates(duplicate_groups, keep_strategy)
                
            return {
                'success': True,
                'message': f'analysis complete. found {len(duplicate_groups)} duplicate groups.',
                'images_scanned': len(images),
                'duplicate_groups': len(duplicate_groups),
                'removed_files': removal_results['removed_files'],
                'total_space_saved': removal_results['total_space_saved'],
                'errors': removal_results['errors'],
                'report': report,
                'dry_run': dry_run
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'duplicate removal failed: {str(e)}',
                'images_scanned': 0,
                'duplicate_groups': 0,
                'removed_files': [],
                'total_space_saved': 0,
                'errors': [str(e)],
                'report': '',
                'dry_run': dry_run
            }
