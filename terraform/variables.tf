variable "region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "vectorforge"
}

variable "openai_api_key" {
  sensitive = true
  default   = ""
}

variable "hf_token" {
  sensitive = true
  default   = ""
}

variable "configs" {
  description = "List of config file paths to run as batch jobs"
  type        = list(string)
  default = [
    "configs/mteb_tweets_openai.yaml",
    "configs/nick007x_arxiv_papers.yaml",
    "configs/openassistant_oasst1.yaml",
  ]
}

variable "enable_gpu" {
  description = "Set to true to create GPU compute environment and job queue"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "EC2 GPU instance types for local model embedding"
  type        = list(string)
  default     = ["g5.xlarge"] # 1x A10G, 24GB VRAM, 4 vCPU, 16GB RAM
}

variable "gpu_max_vcpus" {
  default = 32
}
