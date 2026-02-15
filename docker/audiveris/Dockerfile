# syntax=docker/dockerfile:1.7

ARG AUDIVERIS_VERSION=5.9.0

FROM eclipse-temurin:25-jdk-jammy AS build
ARG AUDIVERIS_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    tar \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${AUDIVERIS_VERSION}" https://github.com/Audiveris/audiveris.git .
RUN ./gradlew --no-daemon clean distTar

RUN set -eux; \
    dist_tar="$(find build/distributions -maxdepth 1 -type f -name '*.tar' | head -n 1)"; \
    test -n "${dist_tar}"; \
    mkdir -p /out; \
    tar -xf "${dist_tar}" -C /out; \
    app_dir="$(find /out -maxdepth 1 -mindepth 1 -type d | head -n 1)"; \
    test -n "${app_dir}"; \
    mv "${app_dir}" /out/audiveris

FROM eclipse-temurin:25-jre-jammy AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    fontconfig \
    fonts-dejavu-core \
    ghostscript \
    libfreetype6 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Audiveris works better when legacy tesseract data is explicitly available.
RUN mkdir -p /opt/tessdata-legacy \
    && curl -fsSL -o /opt/tessdata-legacy/eng.traineddata \
      https://github.com/tesseract-ocr/tessdata/raw/3.04.00/eng.traineddata

COPY --from=build /out/audiveris /opt/audiveris
COPY docker/audiveris/print-version.sh /usr/local/bin/audiveris-version

RUN chmod +x /opt/audiveris/bin/Audiveris /usr/local/bin/audiveris-version

ENV AUDIVERIS_HOME=/opt/audiveris
ENV PATH="/opt/audiveris/bin:${PATH}"
ENV TESSDATA_PREFIX=/opt/tessdata-legacy
ENV JAVA_TOOL_OPTIONS="-Djava.awt.headless=true"

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["Audiveris -help"]
