"""
Create minimal COCO format annotations from CARLA data splits.
"""
import json
from pathlib import Path

def create_coco_annotations(split_type='train'):
    """Create minimal COCO format annotations for a data split."""
    
    # Load split indices
    splits_dir = Path('./splits')
    split_file = splits_dir / f'{split_type}_indices.json'
    
    if not split_file.exists():
        print(f"Error: {split_file} not found")
        return None
    
    with open(split_file, 'r') as f:
        split_indices = json.load(f)
    
    # Create COCO format structure with minimal required fields
    coco_dict = {
        'info': {
            'description': f'CARLA Stereo Pedestrian {split_type.capitalize()} Split',
            'version': '1.0',
            'year': 2025,
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'person', 'supercategory': 'person'}
        ]
    }
    
    # Add dummy entries for each frame in split
    # Real keypoints will come from the custom dataset loader during training
    for idx_entry in split_indices[:200]:  # Limit to first 200 for testing
        global_idx = idx_entry['global_index']
        image_id = global_idx
        annotation_id = global_idx
        
        # Create COCO image entry
        image_entry = {
            'id': image_id,
            'file_name': f'carla_{global_idx:06d}.jpg',
            'height': 512,
            'width': 512,
        }
        coco_dict['images'].append(image_entry)
        
        # Create minimal annotation entry (real keypoints will be added by custom dataset)
        annotation_entry = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,
            'bbox': [10, 10, 100, 200],  # Dummy bbox
            'area': 20000,
            'keypoints': [0] * 51,  # 17 keypoints * 3 (x, y, visibility)
            'num_keypoints': 0,
            'iscrowd': 0,
        }
        coco_dict['annotations'].append(annotation_entry)
    
    # Save COCO annotations
    output_dir = Path('./coco_annotations')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f'{split_type}_annotations.json'
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    
    num_images = len(coco_dict['images'])
    print(f"Created {output_file} with {num_images} images")
    return str(output_file)

if __name__ == '__main__':
    # Create annotations for all splits
    for split in ['train', 'test', 'eval']:
        print(f"\n=== Creating {split} annotations ===")
        create_coco_annotations(split)
