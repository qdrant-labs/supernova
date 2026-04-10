variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 GPU instance type"
  type        = string
  default     = "g5.xlarge" # A10G — also try g4dn.xlarge (T4), g6.xlarge (L4)
}

# use like: terraform apply -var="key_pair_name=my-key"
variable "key_pair_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "volume_size" {
  description = "Root EBS volume size in GiB (need space for models + CUDA)"
  type        = number
  default     = 100
}
