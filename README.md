# Amica Stack

AWS infrastructure boilerplate for deploying Chainlit applications using ECS Fargate with HTTPS support.

**⚠️ Disclaimer**: This is infrastructure boilerplate only. Additional work is required to deploy a complete RAG application, including vector databases, document processing, AI model integration, and production security enhancements.

## Quick Setup

### Prerequisites

- AWS CLI configured for ap-southeast-1
- Node.js 18+ and npm  
- Docker
- AWS CDK CLI: `npm install -g aws-cdk`

### 1. Configure Account ID

Edit `cdk/bin/cdk.ts` and replace `074797805133` with your AWS account ID.

### 2. Bootstrap CDK (first time only)

```bash
cd cdk
cdk bootstrap aws://YOUR-ACCOUNT-ID/ap-southeast-1
```

### 3. Install Dependencies

```bash
cd cdk
npm install
cd ..
```

### 4. Deploy Infrastructure

```bash
cd cdk
cdk deploy --region ap-southeast-1
```

Note the outputs: `RepositoryUri` and `CloudFrontDomainName`.

### 5. Build and Push Application

```bash
# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Build and push
docker build -t amica-stack .
docker tag amica-stack:latest YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com/amica-stack:latest
docker push YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com/amica-stack:latest
```

### 6. Deploy Application

```bash
aws ecs update-service --cluster amica-cluster --service amica-service --force-new-deployment --region ap-southeast-1
```

Your application will be available at the CloudFront domain from step 4.

## Local Development

```bash
uv sync
chainlit run app.py
```

Visit `http://localhost:8000` to test locally.

## Updates

### Update Application Code
1. Modify `app.py`
2. Repeat steps 5-6 above

### Update Infrastructure
1. Modify files in `cdk/lib/`
2. Run `cdk deploy --region ap-southeast-1`

## Cleanup

```bash
cd cdk
cdk destroy --region ap-southeast-1
```

**Note**: CloudFront distribution may take 10-15 minutes to fully delete.

## Project Structure

```
├── app.py              # Chainlit application
├── Dockerfile          # Container configuration  
├── pyproject.toml      # Python dependencies
├── cdk/                # AWS CDK infrastructure
└── docs/               # Detailed documentation
    └── DEPLOYMENT.md   # Architecture and configuration details
```

## Documentation

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed architecture explanation, cost analysis, security considerations, and advanced configuration options. 