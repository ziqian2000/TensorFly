apt-get install gfortran
git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
make PREFIX=/usr/local/Openblas install
cp -f /usr/local/Openblas/include/* /usr/local/include/
rm -f /usr/local/lib/libopenblas.a
rm -f /usr/local/lib/libopenblas.so
ln -f -s /usr/local/Openblas/lib/libopenblas_haswellp-r0.3.3.dev.a /usr/local/lib/libopenblas.a
ln -f -s /usr/local/Openblas/lib/libopenblas.so.0 /usr/local/lib/libopenblas.so
