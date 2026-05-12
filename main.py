from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_matrix(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            n = int(lines[0].strip())
            matrix = []
            for line in lines[1:]:
                matrix.extend([float(x) for x in line.split()])
        return n, matrix
    except FileNotFoundError:
        print(f"Hata: {filename} dosyasi bulunamadi!")
        comm.Abort(1) # Programı güvenli şekilde sonlandır

n = None
a = None
b = None

if rank == 0:
    n, a = read_matrix('a.txt')
    _, b = read_matrix('b.txt')
    start_time = time.time()
else:
    a = None
    b = None

# N değerini ve B matrisini herkese gönder
n = comm.bcast(n, root=0)
b = comm.bcast(b, root=0)

# A matrisini satır satır dağıt
rows_per_proc = n // size
sub_a = comm.scatter([a[i*rows_per_proc*n : (i+1)*rows_per_proc*n] for i in range(size)] if rank == 0 else None, root=0)

# Hesaplama
sub_c = [0] * (rows_per_proc * n)
for i in range(rows_per_proc):
    for j in range(n):
        for k in range(n):
            sub_c[i * n + j] += sub_a[i * n + k] * b[k * n + j]

# Sonuçları topla
res = comm.gather(sub_c, root=0)

if rank == 0:
    end_time = time.time()
    # Sonuçları tek bir liste haline getir
    c = [item for sublist in res for item in sublist]
    print(f"Python - N={n}, Proc={size}, Süre={end_time - start_time:.6f} sn")
