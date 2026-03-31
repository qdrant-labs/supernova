variable "region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "vectorforge"
}

variable "openai_api_key" {
  sensitive = true
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
