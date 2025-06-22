# Amica Stack

A production-ready Chainlit conversational AI application deployed on AWS with HTTPS support.

## Overview

This project demonstrates how to deploy a Chainlit chat application to AWS using:
- **ECS Fargate** for containerized application hosting
- **Application Load Balancer** for reliable traffic distribution  
- **CloudFront** for global HTTPS delivery
- **AWS CDK** for Infrastructure as Code

**Live Demo**: Chat interface that greets users and echoes messages with deployment context.

## Architecture

```
Internet → CloudFront → ALB → ECS Fargate
```

- **Cost**: ~$25-33/month
- **Region**: ap-southeast-1 (Singapore)
- **Features**: Auto-scaling, health checks, HTTPS, global CDN

## Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js 18+ and npm
- Docker
- AWS CDK CLI: `npm install -g aws-cdk`

## Quick Start

1. **Clone and Install**
   ```bash
   git clone <repository-url>
   cd amica-stack
   ```

2. **Bootstrap CDK** (first time only)
   ```bash
   cd cdk
   cdk bootstrap aws://YOUR-ACCOUNT-ID/ap-southeast-1
   ```

3. **Build and Push Container**
   ```bash
   # Get ECR login
   aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com
   
   # Build and push
   docker build --platform linux/amd64 -t amica-stack .
   docker tag amica-stack:latest YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com/amica-stack:latest
   docker push YOUR-ACCOUNT-ID.dkr.ecr.ap-southeast-1.amazonaws.com/amica-stack:latest
   ```

4. **Deploy Infrastructure**
   ```bash
   cd cdk
   npm install
   npm run build
   cdk deploy --require-approval never
   ```

5. **Access Your App**
   - Your application will be available at the CloudFront URL shown in the deployment output
   - Format: `https://xxxxx.cloudfront.net`

## Management

### View Deployment Status
```bash
cd cdk
cdk list
aws cloudformation describe-stacks --stack-name AmicaStackStack --region ap-southeast-1
```

### Update Application
```bash
# Rebuild and push container (steps 3 above)
# Infrastructure updates automatically
```

### Clean Up
```bash
cd cdk
cdk destroy --force
```

**Note**: CloudFront distributions may take 10-15 minutes to fully delete.

## Project Structure

```
├── app.py              # Chainlit application
├── Dockerfile          # Container configuration  
├── pyproject.toml      # Python dependencies
├── cdk/                # AWS CDK infrastructure
└── docs/               # Detailed documentation
    └── DEPLOYMENT.md   # Complete deployment guide
```

## Documentation

- **[Complete Deployment Guide](docs/DEPLOYMENT.md)** - Detailed architecture, costs, and options
- **[AWS CDK Docs](https://docs.aws.amazon.com/cdk/)** - CDK documentation
- **[Chainlit Docs](https://docs.chainlit.io/)** - Chainlit framework documentation

## Local Development

Test the application locally:
```bash
uv run chainlit run app.py
```

Then visit `http://localhost:8000` to test the chat interface. 