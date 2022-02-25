# stage 1
FROM nvidia/cuda:11.2.0-devel-ubuntu18.04 as build

RUN apt update && apt install -y --no-install-recommends \
  curl

RUN curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -

RUN apt install -y --no-install-recommends \
  cmake libvdpau-dev && \
  rm -rf /var/lib/apt/lists/*

COPY cuda-control.tar /tmp

ARG version

RUN cd /tmp && tar xvf /tmp/cuda-control.tar && \
    cd /tmp/cuda-control && mkdir anycuda-${version} && \
    cd anycuda-${version} && cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make

RUN cd /tmp/cuda-control && tar cf /tmp/anycuda.tar.gz -c anycuda-${version}

# stage 2
FROM centos:7 as rpmpkg

RUN yum install -y rpm-build
RUN mkdir -p /root/rpmbuild/{SPECS,SOURCES}

COPY anycuda.spec /root/rpmbuild/SPECS
COPY --from=build /tmp/anycuda.tar.gz /root/rpmbuild/SOURCES

RUN echo '%_topdir /root/rpmbuild' > /root/.rpmmacros \
  && echo '%__os_install_post %{nil}' >> /root/.rpmmacros \
  && echo '%debug_package %{nil}' >> /root/.rpmmacros

WORKDIR /root/rpmbuild/SPECS

ARG version
ARG commit

RUN rpmbuild -bb --quiet \
  --define 'version '${version}'' \
  --define 'commit '${commit}'' \
  anycuda.spec

# stage 3
FROM centos:7

ARG version
ARG commit

COPY --from=rpmpkg  /root/rpmbuild/RPMS/x86_64/anycuda-${version}-${commit}.el7.x86_64.rpm /tmp
RUN rpm -ivh /tmp/anycuda-${version}-${commit}.el7.x86_64.rpm && rm -rf /tmp/anycuda-${version}-${commit}.el7.x86_64.rpm
