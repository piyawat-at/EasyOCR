FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

# if you forked EasyOCR, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
ARG service_home="/home/EasyOCR"
ARG gh_username=piyawat-at
# Configure apt and install packages
RUN apt-get update -y && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    unzip \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists

# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && git clone "https://github.com/$gh_username/EasyOCR.git" "$service_home" \
    # && git config --global user.email $gh_email \
    # && git config --global user.name $gh_username \
    && cd "$service_home" \
    && git remote add upstream "https://github.com/JaidedAI/EasyOCR.git" \
    && git pull upstream master


# Build
RUN cd "$service_home" \
    && python setup.py build_ext --inplace -j 4 \
    && python -m pip install -e .
# Load model folder
WORKDIR $service_home
RUN mkdir -p models
WORKDIR $service_home/models
RUN gdown 'https://drive.google.com/uc?id=1I0bLEXc51FZ9Nk86-FjVn1NFw2vwz7RA'
RUN unzip 'models.zip'
RUN rm 'models.zip'
WORKDIR $service_home
