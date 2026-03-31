output "ecr_repository_url" {
  value = aws_ecr_repository.vectorforge.repository_url
}

output "job_queue_arn" {
  value = aws_batch_job_queue.vectorforge.arn
}

output "job_definition_arns" {
  value = { for k, v in aws_batch_job_definition.per_config : k => v.arn }
}
