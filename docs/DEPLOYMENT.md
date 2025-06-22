# Amica Stack - Architecture and Configuration Guide

This document provides detailed information about the Amica Stack architecture, AWS services, configuration options, and operational considerations.

## Stack Overview

The Amica Stack is a comprehensive AWS infrastructure solution for deploying Chainlit conversational AI applications. It demonstrates modern cloud-native patterns using fully managed AWS services with Infrastructure as Code principles.

### Current Application State

**Demo Application**: The included `app.py` is a simple echo application that validates the infrastructure setup. It demonstrates WebSocket connectivity and basic Chainlit functionality.

**RAG Ready**: While the current application is minimal, the infrastructure is designed to support full RAG (Retrieval-Augmented Generation) implementations with vector databases, document processing, and AI model integration. Resource adjustments will be needed for production RAG workloads.

## Architecture Deep Dive

### Service Architecture

```
Internet → CloudFront → Application Load Balancer → ECS Fargate Tasks
                                ↓
                        CloudWatch Logs & Metrics
```

**CloudFront Distribution**
- Global content delivery network with 200+ edge locations
- Automatic HTTPS certificate provisioning and management
- WebSocket support for real-time Chainlit interactions
- Origin failover and caching optimizations
- DDoS protection through AWS Shield Standard

**Application Load Balancer (ALB)**
- Layer 7 load balancing with health check validation
- SSL termination and HTTP to HTTPS redirection
- Target group management for zero-downtime deployments
- Connection draining for graceful task replacement
- Integration with ECS service discovery

**ECS Fargate**
- Serverless container orchestration without EC2 management
- Automatic task replacement on health check failures
- Built-in service mesh capabilities through AWS App Mesh (optional)
- Task definition versioning for rollback capabilities
- Integration with AWS Systems Manager for configuration

**Amazon ECR (Elastic Container Registry)**
- Private Docker registry with encryption at rest
- Vulnerability scanning for container images
- Lifecycle policies for automated image cleanup
- Cross-region replication capabilities
- Integration with CI/CD pipelines

### Network Architecture

**VPC Configuration**
- Custom VPC with CIDR block 10.0.0.0/16
- Public subnets across multiple Availability Zones
- Internet Gateway for direct internet access
- Route tables configured for high availability

**Security Groups**
- ALB security group: Allows HTTP/HTTPS from anywhere (0.0.0.0/0)
- ECS security group: Restricts port 8080 access to ALB only
- Principle of least privilege with minimal required permissions
- Stateful firewall rules with automatic return traffic handling

**Current Network Model**: Development-optimized with public subnets for cost efficiency and simplified networking.

## Resource Configuration Analysis

### Current Allocation (Development Optimized)

**CPU and Memory**
- **CPU**: 256 CPU units (0.25 vCPU)
- **Memory**: 512 MB RAM
- **Storage**: 20 GB ephemeral storage (default)
- **Network**: Up to 10 Gbps networking performance

**Scaling Configuration**
- **Desired Count**: 1 task
- **Minimum Capacity**: 1 task
- **Maximum Capacity**: 1 task (manual scaling)
- **Deployment Configuration**: Rolling update with 100% minimum healthy percent

### Resource Recommendations for Different Workloads

**RAG Applications (Recommended Upgrade)**: For production RAG workloads, upgrade to at least 1024 CPU units (1+ vCPU) and 2048+ MB memory to handle vector computations, model loading, and embeddings processing. Consider 30+ GB storage for model caching and document processing, with 2-4 tasks running auto-scaling policies to handle variable workloads efficiently.

**High-Traffic Production**: Enterprise applications should provision 2048+ CPU units (2+ vCPU) and 4096+ MB memory per task, scaling to 5-10 tasks with CloudWatch-based auto-scaling policies. Implement multiple target groups for A/B testing and advanced deployment strategies to ensure high availability and performance under load.

### Performance Characteristics

**Cold Start**: ~30-60 seconds for initial task startup
**Warm Performance**: <100ms response times for simple operations
**Scaling Events**: ~2-3 minutes for new task provisioning
**Health Check**: 30-second intervals with 5-second timeout

## Cost Analysis and Optimization

### Detailed Cost Breakdown (ap-southeast-1)

**ECS Fargate Pricing**
- **vCPU**: $0.04048 per vCPU-hour
- **Memory**: $0.004445 per GB-hour
- **Current Configuration**: 0.25 vCPU × 0.5 GB × 730 hours = ~$4-5/month

**Application Load Balancer**
- **Fixed Cost**: $16.20/month (always running)
- **LCU Usage**: ~$1-2/month for development traffic
- **Total ALB**: ~$16-18/month

**CloudFront Distribution**
- **Requests**: $0.0075 per 10,000 HTTP requests
- **Data Transfer**: $0.085 per GB (first 10 TB)
- **Development Usage**: ~$1-3/month

**Supporting Services**
- **ECR Storage**: $0.10 per GB-month (~$1/month for few images)
- **CloudWatch Logs**: $0.50 per GB ingested (~$1-2/month)
- **VPC**: No additional charges for public subnet configuration

**Total Monthly Cost**: ~$23-29/month for development environment

### Cost Scaling Projections

**Production RAG Application** (~2-4 tasks, 1+ vCPU each)
- ECS Fargate: ~$35-70/month
- ALB: ~$16-18/month
- CloudFront: ~$5-15/month (higher traffic)
- **Total**: ~$56-103/month

**Enterprise Scale** (10+ tasks, auto-scaling)
- ECS Fargate: ~$150-300/month
- Enhanced monitoring and logging: ~$20-40/month
- Additional services (RDS, OpenSearch): ~$100-200/month
- **Total**: ~$270-540/month

## Security Architecture

### Current Security Model

**Network Security**
- ECS tasks deployed in public subnets with public IP addresses
- Direct internet access for external API calls and dependency downloads
- ALB provides SSL termination and HTTP to HTTPS redirection
- Security groups implement network-level access controls

**Application Security**
- HTTPS enforcement through CloudFront with AWS-managed certificates
- Automatic certificate renewal and deployment
- HTTP/2 and TLS 1.2+ encryption standards
- Basic DDoS protection through AWS Shield Standard

**IAM and Access Control**
- Task execution role with minimal required permissions
- ECR access for container image pulls
- CloudWatch Logs access for application logging
- No unnecessary service permissions or overly broad policies

### Security Enhancement Roadmap

**Network Improvements**
- **Private Subnets**: Move ECS tasks to private subnets for enhanced isolation
- **NAT Gateway**: Implement NAT Gateway for secure outbound internet access
- **VPC Endpoints**: Add VPC endpoints for AWS service access without internet routing
- **Network ACLs**: Implement subnet-level access controls for defense in depth

**Application Security Enhancements**
- **AWS WAF**: Deploy Web Application Firewall for CloudFront protection
- **API Rate Limiting**: Implement request throttling and abuse prevention
- **Secrets Management**: Integrate AWS Secrets Manager for API keys and credentials
- **Input Validation**: Add comprehensive input sanitization and validation layers

**Monitoring and Compliance**
- **VPC Flow Logs**: Enable network traffic monitoring and analysis
- **AWS Config**: Implement configuration compliance monitoring
- **CloudTrail**: Enable API call logging and audit trails
- **AWS Security Hub**: Centralized security finding aggregation

## RAG Application Integration

### Infrastructure Requirements for RAG

**Vector Database Integration**
- **Amazon OpenSearch**: Managed Elasticsearch with vector search capabilities
- **Pinecone**: External vector database with API integration
- **pgvector**: PostgreSQL extension for vector operations in RDS

**Document Storage and Processing**
- **Amazon S3**: Document storage with lifecycle policies
- **AWS Lambda**: Serverless document processing and chunking
- **Amazon Textract**: OCR and document analysis for PDFs and images

**AI Model Integration**
- **Amazon Bedrock**: Managed access to foundation models
- **SageMaker**: Custom model hosting and inference endpoints
- **External APIs**: OpenAI, Anthropic, Cohere integration through secrets management

### Environment Configuration

**Required Environment Variables for RAG**
```bash
# AI Service Configuration
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
COHERE_API_KEY=xxx

# Vector Database
PINECONE_API_KEY=xxx
PINECONE_ENVIRONMENT=us-west1-gcp
OPENSEARCH_ENDPOINT=https://xxx.amazonaws.com
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=xxx

# Document Storage
S3_BUCKET_NAME=amica-documents
S3_REGION=ap-southeast-1

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://cache:6379/0
```

**Enhanced Dependencies for RAG**
```toml
dependencies = [
    "chainlit>=2.3.0",
    "openai>=1.0.0",
    "anthropic>=0.8.0",
    "langchain>=0.1.0",
    "sentence-transformers>=2.2.0",
    "pinecone-client>=3.0.0",
    "opensearch-py>=2.4.0",
    "boto3>=1.34.0",
    "psycopg2-binary>=2.9.0",
    "redis>=5.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]
```

## Monitoring and Observability

### Available Monitoring

**CloudWatch Integration**
- **Application Logs**: Centralized in `/ecs/amica-stack` log group
- **ECS Metrics**: CPU, memory, network, and disk utilization
- **ALB Metrics**: Request count, latency, error rates, and target health
- **CloudFront Metrics**: Cache hit ratio, origin latency, and error rates

**Health Check Configuration**
- **Health Check Path**: `/health` (configurable)
- **Check Interval**: 30 seconds
- **Timeout**: 5 seconds
- **Healthy Threshold**: 2 consecutive successes
- **Unhealthy Threshold**: 2 consecutive failures

### Production Monitoring Enhancements

**Application Performance Monitoring (APM)**
- AWS X-Ray for distributed tracing
- Custom metrics for business KPIs
- Performance profiling and bottleneck identification

**Alerting and Notifications**
- CloudWatch Alarms for critical metrics
- SNS integration for email/SMS notifications
- PagerDuty or OpsGenie integration for incident response

**Log Analysis and SIEM**
- Amazon OpenSearch for log aggregation
- AWS Security Hub for security findings
- Custom dashboards for operational insights

## Infrastructure Management

### CDK Benefits and Patterns

**Infrastructure as Code Advantages**
- **Version Control**: Infrastructure changes tracked with application code
- **Reproducible Deployments**: Consistent environments across stages
- **Type Safety**: TypeScript provides compile-time validation
- **Automated Testing**: Unit tests for infrastructure components

**Deployment Patterns**
- **Blue-Green Deployments**: Zero-downtime updates with traffic shifting
- **Rolling Updates**: Gradual replacement of tasks with health validation
- **Canary Releases**: Limited exposure testing with metrics validation

### Operational Procedures

**Troubleshooting Commands**

Check ECS service health:
```bash
aws ecs describe-services --cluster amica-cluster --services amica-service --region ap-southeast-1
```

Monitor task events:
```bash
aws ecs describe-tasks --cluster amica-cluster --tasks $(aws ecs list-tasks --cluster amica-cluster --service-name amica-service --query 'taskArns[0]' --output text) --region ap-southeast-1
```

View application logs:
```bash
aws logs get-log-events --log-group-name /ecs/amica-stack --log-stream-name $(aws logs describe-log-streams --log-group-name /ecs/amica-stack --query 'logStreams[0].logStreamName' --output text) --region ap-southeast-1
```

Check ALB target health:
```bash
aws elbv2 describe-target-health --target-group-arn $(aws elbv2 describe-target-groups --names amica-tg --query 'TargetGroups[0].TargetGroupArn' --output text) --region ap-southeast-1
```

**Performance Optimization**
- Monitor CloudWatch metrics for resource utilization
- Analyze ALB access logs for traffic patterns
- Review CloudFront cache performance and hit ratios
- Optimize container image size and startup time

## Development and Deployment Workflow

### Local Development Setup

**Development Environment**
```bash
# Install Python dependencies
uv sync

# Run application locally
chainlit run app.py --host 0.0.0.0 --port 8000

# Test WebSocket connectivity
# Visit http://localhost:8000 for chat interface
```

**Container Testing**
```bash
# Build and test container locally
docker build -t amica-stack .
docker run -p 8080:8080 -e CHAINLIT_HOST=0.0.0.0 amica-stack

# Test container health endpoint
curl http://localhost:8080/health
```

### CI/CD Integration Patterns

**GitHub Actions Integration**
```yaml
# Example workflow for automated deployment
- name: Deploy to AWS
  run: |
    cdk deploy --require-approval never
    docker build -t amica-stack .
    docker tag amica-stack:latest $ECR_REPOSITORY:latest
    docker push $ECR_REPOSITORY:latest
    aws ecs update-service --cluster amica-cluster --service amica-service --force-new-deployment
```

**Infrastructure Testing**
- CDK unit tests for resource validation
- Integration tests for service connectivity
- Security scanning with AWS Inspector or third-party tools
- Performance testing with load simulation

This architecture provides a solid foundation for conversational AI applications while maintaining flexibility for future enhancements and production-grade requirements.
