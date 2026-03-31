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
