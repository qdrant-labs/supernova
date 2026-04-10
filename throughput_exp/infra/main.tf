terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# detect operator IP at apply time
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

locals {
  my_ip = "${trimspace(data.http.my_ip.response_body)}/32"
  tags  = { Project = "vectorforge-throughput-exp" }
}

# AWS DLAMI — NVIDIA drivers + CUDA + PyTorch pre-installed
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# networking
data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "exp" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "throughput-exp-vpc" })
}

resource "aws_subnet" "exp" {
  vpc_id                  = aws_vpc.exp.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "throughput-exp-subnet" })
}

resource "aws_internet_gateway" "exp" {
  vpc_id = aws_vpc.exp.id
  tags   = merge(local.tags, { Name = "throughput-exp-igw" })
}

resource "aws_route_table" "exp" {
  vpc_id = aws_vpc.exp.id
  tags   = merge(local.tags, { Name = "throughput-exp-rt" })

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.exp.id
  }
}

resource "aws_route_table_association" "exp" {
  subnet_id      = aws_subnet.exp.id
  route_table_id = aws_route_table.exp.id
}

# security group — SSH only, locked to operator IP
resource "aws_security_group" "exp" {
  name_prefix = "throughput-exp-"
  description = "SSH from operator IP only"
  vpc_id      = aws_vpc.exp.id
  tags        = local.tags

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.my_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# gpu instance
resource "aws_instance" "gpu" {
  ami                         = data.aws_ami.dlami.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.exp.id
  vpc_security_group_ids      = [aws_security_group.exp.id]
  associate_public_ip_address = true
  monitoring                  = true
  key_name                    = var.key_pair_name

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
  }

  user_data = file("${path.module}/user_data/setup.sh")

  tags = merge(local.tags, { Name = "throughput-exp-${var.instance_type}" })
}
