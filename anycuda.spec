Name: anycuda
Version: %{version}
Release: %{commit}%{?dist}
Summary: GPU virtual device library

License: MIT
Source: anycuda.tar.gz

Requires: systemd-units

%define pkgname %{name}-%{version}-%{release}

%description
GPU virtual device library

%prep
%setup

%install
install -d $RPM_BUILD_ROOT/%{_libdir}
install -d $RPM_BUILD_ROOT/%{_bindir}

install -p -m 755 libcuda-control.so $RPM_BUILD_ROOT/%{_libdir}/

%clean
rm -rf $RPM_BUILD_ROOT

%files
/%{_libdir}/libcuda-control.so

%post
ldconfig

%postun
ldconfig
