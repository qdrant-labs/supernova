terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ecr repository for batch job images

resource "aws_ecr_repository" "vectorforge" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# iam role
data "aws_iam_policy_document" "batch_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_job" {
  name               = "${var.project_name}-batch-job"
  assume_role_policy = data.aws_iam_policy_document.batch_assume.json
}

# S3 full access for uploading parquets
resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.batch_job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# CloudWatch logs
resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.batch_job.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# ECR pull
resource "aws_iam_role_policy_attachment" "ecr_pull" {
  role       = aws_iam_role.batch_job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# batch compute environment

resource "aws_batch_compute_environment" "vectorforge" {
  name = var.project_name
  type                     = "MANAGED"
  state                    = "ENABLED"

  compute_resources {
    type      = "FARGATE"
    max_vcpus = 16

    subnets            = data.aws_subnets.default.ids
    security_group_ids = [data.aws_security_group.default.id]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_security_group" "default" {
  vpc_id = data.aws_vpc.default.id
  name   = "default"
}

# batch job queue
resource "aws_batch_job_queue" "vectorforge" {
  name     = var.project_name
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.vectorforge.arn
  }
}

# batch job definition
resource "aws_cloudwatch_log_group" "vectorforge" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = 14
}

resource "aws_batch_job_definition" "vectorforge" {
  name = var.project_name
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image = "${aws_ecr_repository.vectorforge.repository_url}:latest"

    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" },
    ]

    jobRoleArn       = aws_iam_role.batch_job.arn
    executionRoleArn = aws_iam_role.batch_job.arn

    environment = [
      { name = "OPENAI_API_KEY", value = var.openai_api_key },
      { name = "HF_TOKEN", value = var.hf_token },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.vectorforge.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "batch"
      }
    }

    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
  })
}

# --- GPU compute environment (EC2-backed, optional) ---

data "aws_iam_policy_document" "batch_service_assume" {
  count = var.enable_gpu ? 1 : 0
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["batch.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "batch_service" {
  count              = var.enable_gpu ? 1 : 0
  name               = "${var.project_name}-batch-service"
  assume_role_policy = data.aws_iam_policy_document.batch_service_assume[0].json
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  count      = var.enable_gpu ? 1 : 0
  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# EC2 instance profile for GPU instances
resource "aws_iam_role" "ecs_instance" {
  count              = var.enable_gpu ? 1 : 0
  name               = "${var.project_name}-ecs-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_instance" {
  count      = var.enable_gpu ? 1 : 0
  role       = aws_iam_role.ecs_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "ecs_instance" {
  count = var.enable_gpu ? 1 : 0
  name  = "${var.project_name}-ecs-instance"
  role  = aws_iam_role.ecs_instance[0].name
}

resource "aws_batch_compute_environment" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${var.project_name}-gpu"
  type  = "MANAGED"
  state = "ENABLED"

  compute_resources {
    type                = "EC2"
    max_vcpus           = var.gpu_max_vcpus
    min_vcpus           = 0
    desired_vcpus       = 0
    instance_type       = var.gpu_instance_types
    instance_role       = aws_iam_instance_profile.ecs_instance[0].arn
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [data.aws_security_group.default.id]
  }
}

resource "aws_batch_job_queue" "gpu" {
  count    = var.enable_gpu ? 1 : 0
  name     = "${var.project_name}-gpu"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu[0].arn
  }
}

resource "aws_batch_job_definition" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${var.project_name}-gpu"
  type  = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image = "${aws_ecr_repository.vectorforge.repository_url}:latest-gpu"

    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "15000" },
      { type = "GPU", value = "1" },
    ]

    jobRoleArn       = aws_iam_role.batch_job.arn
    executionRoleArn = aws_iam_role.batch_job.arn

    environment = [
      { name = "OPENAI_API_KEY", value = var.openai_api_key },
      { name = "HF_TOKEN", value = var.hf_token },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.vectorforge.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "batch-gpu"
      }
    }
  })
}

# submit one job definition per config file, with the config file path as the command argument
resource "aws_batch_job_definition" "per_config" {
  for_each = toset(var.configs)

  name = "${var.project_name}-${replace(replace(each.value, "/", "-"), ".yaml", "")}"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image   = "${aws_ecr_repository.vectorforge.repository_url}:latest"
    command = [each.value]

    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" },
    ]

    jobRoleArn       = aws_iam_role.batch_job.arn
    executionRoleArn = aws_iam_role.batch_job.arn

    environment = [
      { name = "OPENAI_API_KEY", value = var.openai_api_key },
      { name = "HF_TOKEN", value = var.hf_token },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.vectorforge.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "batch"
      }
    }

    networkConfiguration = {
      assignPublicIp = "ENABLED"
    }
  })
}
