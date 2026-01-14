import shutil
import os
import signal

def trigger_rewind():
    print("🔄 REWINDING: Replacing fine-tuned weights with Gold Standard...")
    
    # 1. Path to weights
    gold_weights = "weights/gold_reflex.pth"
    active_weights = "weights/active_reflex.pth"
    
    # 2. Overwrite the bad fine-tuned weights
    if os.path.exists(gold_weights):
        shutil.copy(gold_weights, active_weights)
        print("✅ Gold weights restored.")
        
        # 3. Optional: Restart the main bridge to reload weights
        # os.kill(os.getppid(), signal.SIGHUP) 
    else:
        print("❌ Error: Gold weights not found!")