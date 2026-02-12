from huggingface_hub import HfApi

api = HfApi()

repo_id = "Svram/finers-iterative-qwen2_5-vl-dense-step9120"
local_model_dir = "ckpts/finers_iterative_class_dense/global_step_9120/actor/huggingface"

# Create repo if needed
api.create_repo(repo_id=repo_id, private=True, exist_ok=True)

# Upload folder
api.upload_folder(
    folder_path=local_model_dir,
    repo_id=repo_id,
    repo_type="model",
)
