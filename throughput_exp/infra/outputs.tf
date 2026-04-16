output "public_ip" {
  description = "GPU instance public IP"
  value       = aws_instance.gpu.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.gpu.id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh ec2-user@${aws_instance.gpu.public_ip}"
}

output "scp_bench" {
  description = "Command to upload bench script"
  value       = "scp ../throughput_bench.py ec2-user@${aws_instance.gpu.public_ip}:~/"
}

output "my_ip" {
  description = "Your detected public IP"
  value       = trimspace(data.http.my_ip.response_body)
}
