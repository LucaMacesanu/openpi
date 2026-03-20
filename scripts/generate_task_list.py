import json
import random
from datasets import load_dataset
from tqdm import tqdm

def get_lerobot_task_list(repo_id="physical-intelligence/libero", seed=42):
    print(f"Streaming dataset {repo_id} to find unique tasks...")
    
    # We use streaming=True to avoid downloading the whole dataset
    ds = load_dataset(repo_id, split="train", streaming=True)
    
    unique_tasks = set()
    
    # We'll scan a large number of episodes to ensure we find all tasks.
    # LIBERO usually has 40-130 tasks depending on the version.
    # We'll check the first 5000 episodes (very fast in streaming mode).
    for i, episode in enumerate(tqdm(ds, total=5000)):
        # In most LeRobot/OpenPI configs, the task is stored in 'task' or 'instruction'
        # Adjust 'task' if your specific dataset uses a different key
        task_name = episode.get('task') or episode.get('instruction')
        
        if task_name:
            unique_tasks.add(task_name)
            
        # Stop early if we haven't found a new task in a while 
        # (Optional, but 5000 is usually enough for all LIBERO suites)
        if i > 5000: 
            break

    task_list = sorted(list(unique_tasks))
    
    # Shuffle the list for your Continual Learning order
    random.seed(seed)
    random.shuffle(task_list)
    
    # Format into a list of dicts for your research tracking
    ordered_tasks = [
        {"train_order": i, "task_instruction": task} 
        for i, task in enumerate(task_list)
    ]
    
    return ordered_tasks

if __name__ == "__main__":
    # Generate the randomized order
    training_schedule = get_lerobot_task_list(seed=42)
    
    # Save it so your training script can read it
    with open("continual_learning_schedule.json", "w") as f:
        json.dump(training_schedule, f, indent=4)
        
    print(f"\nFound {len(training_schedule)} unique tasks.")
    for task in training_schedule[:5]:
        print(f"Step {task['train_order']}: {task['task_instruction']}")