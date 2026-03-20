import json
import random
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

def get_lerobot_task_list(repo_id="physical-intelligence/libero", seed=42):
    print(f"Loading metadata for {repo_id} via LeRobotDataset...")
    
    # This specifically loads the dataset using LeRobot's logic
    # It handles the 'List' vs 'Sequence' metadata mismatch internally
    dataset = LeRobotDataset(repo_id)
    
    unique_tasks = set()
    
    # LeRobot datasets have a 'meta' attribute or we can iterate through episodes
    # We only need to check one frame per episode to get the task string
    print("Extracting unique task instructions...")
    for i in tqdm(range(dataset.num_episodes)):
        # Get the first frame of each episode to find the instruction
        # LeRobot stores the language instruction in the 'task' key
        episode_data = dataset.get_item(dataset.episode_data_index[i])
        task_name = episode_data.get('task')
        
        if task_name:
            unique_tasks.add(task_name)

    task_list = sorted(list(unique_tasks))
    
    # Shuffle the list for your Continual Learning order
    random.seed(seed)
    random.shuffle(task_list)
    
    ordered_tasks = [
        {"train_order": i, "task_instruction": task} 
        for i, task in enumerate(task_list)
    ]
    
    return ordered_tasks

if __name__ == "__main__":
    try:
        training_schedule = get_lerobot_task_list(seed=42)
        
        with open("continual_learning_schedule.json", "w") as f:
            json.dump(training_schedule, f, indent=4)
            
        print(f"\nSuccess! Found {len(training_schedule)} unique tasks.")
        for task in training_schedule[:5]:
            print(f"  Step {task['train_order']}: {task['task_instruction']}")
            
    except Exception as e:
        print(f"\nLeRobotDataset failed with: {e}")
        print("Falling back to manual Pip upgrade...")