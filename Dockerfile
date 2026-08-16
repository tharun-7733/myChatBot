FROM vllm/vllm-openai:v0.5.4

# HF Spaces run as a non-root user, so we set up a user 'user'
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Set the working directory
WORKDIR $HOME/app

# Copy the app files
COPY --chown=user . $HOME/app

# Install FastAPI dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the HF space port
EXPOSE 7860

# Run the startup script
CMD ["bash", "run_space.sh"]