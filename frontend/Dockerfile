# Frontend Dockerfile - Multi-stage build
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Remove any .env files that might have been copied (safety measure)
RUN rm -f .env .env.* 2>/dev/null || true

# Build the application with empty VITE_API_URL (uses relative URLs via nginx proxy)
ENV VITE_API_URL=""
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
