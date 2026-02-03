#!/usr/bin/env python3
"""
Automated Pretrained Weights Downloader
Downloads pretrained weights for ST-GCN, CTR-GCN, and Hyperformer models.

Usage:
    python models/download_pretrained.py --model all
    python models/download_pretrained.py --model ctrgcn
    python models/download_pretrained.py --model hyperformer
"""

import os
import sys
import argparse
import subprocess
import urllib.request
from pathlib import Path
import zipfile
import shutil

class PretrainedDownloader:
    def __init__(self, download_dir="pretrained_weights"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        print(f"📁 Download directory: {self.download_dir.absolute()}\n")
    
    def download_hyperformer(self):
        """Download Hyperformer pretrained weights"""
        print("=" * 60)
        print("🚀 Downloading Hyperformer Pretrained Weights")
        print("=" * 60)
        
        hyperformer_dir = self.download_dir / "hyperformer"
        hyperformer_dir.mkdir(exist_ok=True)
        
        # Direct download URL
        url = "https://github.com/ZhouYuxuanYX/Hyperformer/releases/download/pretrained_weights/hyperformer_pretrained_weights.zip"
        zip_path = hyperformer_dir / "hyperformer_pretrained_weights.zip"
        
        # Check if already downloaded
        expected_files = [
            "Hyperformer_ntu60_xsub_joint.pth",
            "Hyperformer_ntu60_xsub_bone.pth"
        ]
        
        if all((hyperformer_dir / f).exists() for f in expected_files):
            print("✅ Hyperformer weights already downloaded!")
            self._list_files(hyperformer_dir)
            return True
        
        try:
            print(f"📥 Downloading from: {url}")
            print("⏳ This may take a few minutes...")
            
            # Download with progress
            urllib.request.urlretrieve(url, zip_path, self._progress_hook)
            print("\n✅ Download complete!")
            
            # Extract
            print(f"📦 Extracting to: {hyperformer_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(hyperformer_dir)
            
            # Cleanup
            zip_path.unlink()
            print("✅ Extraction complete!")
            
            self._list_files(hyperformer_dir)
            return True
            
        except Exception as e:
            print(f"❌ Error downloading Hyperformer: {e}")
            return False
    
    def download_ctrgcn(self):
        """Clone CTR-GCN repository (user needs to get weights from their source)"""
        print("=" * 60)
        print("🔍 Setting up CTR-GCN")
        print("=" * 60)
        
        ctrgcn_dir = self.download_dir / "CTR-GCN"
        
        if ctrgcn_dir.exists():
            print("✅ CTR-GCN repository already cloned!")
            print(f"📁 Location: {ctrgcn_dir.absolute()}")
        else:
            try:
                print("📥 Cloning CTR-GCN repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/Uason-Chen/CTR-GCN.git", str(ctrgcn_dir)],
                    check=True
                )
                print("✅ Repository cloned successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error cloning repository: {e}")
                return False
            except FileNotFoundError:
                print("❌ Git not found. Please install git or clone manually:")
                print("   git clone https://github.com/Uason-Chen/CTR-GCN.git pretrained_weights/CTR-GCN")
                return False
        
        print("\n" + "=" * 60)
        print("⚠️  IMPORTANT: CTR-GCN Pretrained Weights")
        print("=" * 60)
        print("The CTR-GCN pretrained weights are not directly downloadable.")
        print("Please follow these steps:\n")
        print("1. Visit: https://github.com/Uason-Chen/CTR-GCN")
        print("2. Check their README for download links (usually Google Drive)")
        print("3. Download files like:")
        print("   - ntu60_xsub_ctrgcn.pt")
        print("   - ntu60_xview_ctrgcn.pt")
        print("4. Place them in:")
        print(f"   {ctrgcn_dir.absolute()}/")
        print("\nAlternative: Check MMAction2 model zoo:")
        print("   https://github.com/open-mmlab/mmaction2")
        print("=" * 60 + "\n")
        
        return True
    
    def download_stgcn(self):
        """Clone ST-GCN repository"""
        print("=" * 60)
        print("📊 Setting up ST-GCN")
        print("=" * 60)
        
        stgcn_dir = self.download_dir / "st-gcn"
        
        if stgcn_dir.exists():
            print("✅ ST-GCN repository already cloned!")
            print(f"📁 Location: {stgcn_dir.absolute()}")
        else:
            try:
                print("📥 Cloning ST-GCN repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/yysijie/st-gcn.git", str(stgcn_dir)],
                    check=True
                )
                print("✅ Repository cloned successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error cloning repository: {e}")
                return False
            except FileNotFoundError:
                print("❌ Git not found. Please install git or clone manually:")
                print("   git clone https://github.com/yysijie/st-gcn.git pretrained_weights/st-gcn")
                return False
        
        print("\n" + "=" * 60)
        print("⚠️  IMPORTANT: ST-GCN Pretrained Weights")
        print("=" * 60)
        print("Please check the ST-GCN repository for pretrained weights:")
        print("1. Visit: https://github.com/yysijie/st-gcn")
        print("2. Check README or model directory")
        print("3. Or download from MMAction2:")
        print("   https://download.openmmlab.com/mmaction/skeleton/stgcn/")
        print("=" * 60 + "\n")
        
        # Try downloading from OpenMMLab
        print("Attempting to download ST-GCN from OpenMMLab...")
        stgcn_url = "https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint_20200826-e8b0f4b5.pth"
        stgcn_file = stgcn_dir / "stgcn_80e_ntu60_xsub_keypoint.pth"
        
        if stgcn_file.exists():
            print("✅ ST-GCN pretrained weights already downloaded!")
            return True
        
        try:
            print(f"📥 Downloading from OpenMMLab...")
            urllib.request.urlretrieve(stgcn_url, stgcn_file, self._progress_hook)
            print("\n✅ ST-GCN weights downloaded successfully!")
            print(f"📁 Saved to: {stgcn_file}")
            return True
        except Exception as e:
            print(f"⚠️  Could not download from OpenMMLab: {e}")
            print("Please download manually from the ST-GCN repository")
            return False
    
    def download_mmaction2_models(self):
        """Download models from MMAction2"""
        print("=" * 60)
        print("🔧 Downloading from MMAction2 Model Zoo")
        print("=" * 60)
        print("\n⚠️  For better MMAction2 support, use the dedicated script:")
        print("   python setup_mmaction2.py")
        print("\nThis will:")
        print("  - Clone the full MMAction2 repository")
        print("  - Download all skeleton model checkpoints")
        print("  - Create a comprehensive usage guide")
        print("\n" + "=" * 60)
        
        response = input("\nWould you like to run setup_mmaction2.py now? (y/n): ")
        if response.lower() in ('y', 'yes'):
            try:
                subprocess.run([sys.executable, "setup_mmaction2.py"], check=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running setup_mmaction2.py: {e}")
                return False
        else:
            print("Skipping MMAction2 setup. Run 'python setup_mmaction2.py' later.")
            return False
    
    def _progress_hook(self, count, block_size, total_size):
        """Show download progress"""
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r⏳ Progress: {percent}% ")
        sys.stdout.flush()
    
    def _list_files(self, directory):
        """List downloaded files"""
        print(f"\n📦 Downloaded files in {directory.name}:")
        for f in sorted(directory.glob("*.pth*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")
    
    def download_all(self):
        """Download all available pretrained weights"""
        print("\n" + "=" * 60)
        print("🎯 DOWNLOADING ALL PRETRAINED WEIGHTS")
        print("=" * 60 + "\n")
        
        results = {
            "Hyperformer": self.download_hyperformer(),
            "CTR-GCN": self.download_ctrgcn(),
            "ST-GCN": self.download_stgcn(),
        }
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        for model, success in results.items():
            status = "✅" if success else "⚠️"
            print(f"{status} {model}")
        
        print("\n📁 All files downloaded to:")
        print(f"   {self.download_dir.absolute()}")
        print("\n" + "=" * 60)
        print("🎉 Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. For CTR-GCN: Follow instructions above to get weights")
        print("2. Test loading: python models/test_pretrained.py")
        print("3. Start training: python train.py --model ctrgcn --pretrained ...")
        print("\n📖 See PRETRAINED_WEIGHTS_GUIDE.md for detailed usage")
        print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights for GCN models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "hyperformer", "ctrgcn", "stgcn", "mmaction2"],
        default="all",
        help="Which model weights to download"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="pretrained_weights",
        help="Directory to download weights to"
    )
    
    args = parser.parse_args()
    
    downloader = PretrainedDownloader(args.download_dir)
    
    if args.model == "all":
        downloader.download_all()
    elif args.model == "hyperformer":
        downloader.download_hyperformer()
    elif args.model == "ctrgcn":
        downloader.download_ctrgcn()
    elif args.model == "stgcn":
        downloader.download_stgcn()
    elif args.model == "mmaction2":
        downloader.download_mmaction2_models()

if __name__ == "__main__":
    main()
