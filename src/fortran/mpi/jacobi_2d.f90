! Solve the Poisson problem
!     u_{xx} + u_{yy} = f(x, y)   x \in \Omega = [0, pi] x [0, pi]
! with 
!     u(0, y) = u(pi, y) = 0
!     u(x, 0) = 2 sin x
!     u(x, pi) = -2 sin x
! and
!     f(x, y) = -20 sin x cos 3 y
! using Jacobi iterations and MPI.  For simplicity we will assume that we 
! will use a uniform discretization.

program jacobi_2d_mpi

    use mpi

    implicit none

    ! Problem paramters
    real(kind=8), parameter :: PI = 3.141592654d0;
    real(kind=8) :: a = 0.d0, b = PI;

    ! MPI variables
    integer :: num_procs, rank, rank_num_points, error, request
    integer :: start_index, end_index
    integer, dimension(MPI_STATUS_SIZE) :: status

    ! Numerical parameters
    integer, parameter :: MAX_ITERATIONS = 2**16
    integer, parameter :: PRINT_INTERVAL = 1000

    ! Work arrays
    real(kind=8), allocatable :: u_old(:, :)
    real(kind=8), allocatable :: u(:, :)
    real(kind=8), allocatable :: f(:, :)

    ! IO
    character(len=12) :: file_name
    integer, parameter :: IO_UNIT = 42

    ! Locals
    integer :: i, j, N
    integer :: num_points
    real(kind=8) :: x, y, dx, dy, tolerance, du_max, du_max_proc

    ! Initialize MPI
    call MPI_init(error)
    call MPI_comm_size(MPI_COMM_WORLD, num_procs, error)
    call MPI_comm_rank(MPI_COMM_WORLD, rank, error)

    ! Discretization
    num_points = 100
    dx = (b - a) / real(num_points + 1, kind=8)
    dy = dx
    tolerance = 0.1d0 * dx**2

    ! Break up into horizontal strips
    rank_num_points = (num_points + num_procs - 1) / num_procs
    start_index = rank * rank_num_points + 1
    end_index = min((rank + 1) * rank_num_points, num_points)
    rank_num_points = end_index - start_index + 1

    ! Diagnostic: tell the user which points will be handled by which task
    print '("Rank ",i2," = (",i6,", ",i6,")")', rank, start_index, end_index

    ! Allocate memory for the arrays
    allocate(u_old(0:num_points + 1, start_index - 1:end_index + 1))
    allocate(u(0:num_points + 1, start_index - 1:end_index + 1))
    allocate(f(0:num_points + 1, start_index - 1:end_index + 1))

    ! Fill in f and initial guess for u
    do i=1, num_points
        x = dx * real(i, kind=8) + a
        do j=start_index, end_index
            y = dy * real(j + start_index + 1, kind=8) + b
            f(i, j) = -20.d0 * sin(x) * cos(3.d0 * y)
            u(i, j) = -20.d0
        end do
    end do

    ! Set boundary values
    ! Set bottom boundary (rank 0)
    if (rank == 0) then
        do i=0, num_points + 1
            x = dx * real(i, kind=8) + a
            u(i, 0) = 2.0 * sin(x)
        end do
    end if

    ! Set left and right edges (all ranks)
    do j=start_index, end_index
        u(0, j) = 0.d0
        u(num_points + 1, j) = 0.d0
    end do

    ! Set top boundary (num_proces - 1 rank)
    if (rank == num_procs - 1) then
        do i=0, num_points + 1
            x = dx * real(i, kind=8) + a
            u(i, num_points + 1) = -2.0 * sin(x)
        end do
    end if

    !* Jacobi Iteration *!
    du_max = 0.d0
    do N=1, MAX_ITERATIONS
        ! Copy internal values into old array
        u_old = u

        ! Communication of halo values
        ! Send data up (right +) tag = 1
        if (rank < num_procs - 1) then
            call MPI_isend(u_old(:, end_index), num_points + 2, MPI_DOUBLE_PRECISION, &
                            rank + 1, 1, MPI_COMM_WORLD, request, error)
        end if
        ! Send data down (left -) tag = 2
        if (rank > 0) then
            call MPI_isend(u_old(:, 1), num_points + 2, MPI_DOUBLE_PRECISION, &
                            rank - 1, 2, MPI_COMM_WORLD, request, error)
        end if

        ! Receive data from above (left -) tag = 2
        if (rank < num_procs - 1) then
            call MPI_recv(u_old(:, end_index + 1), num_points + 2, MPI_DOUBLE_PRECISION, &
                            rank + 1, 2, MPI_COMM_WORLD, status, error)
        end if
        ! Recieve data from below (right +) tag = 1
        if (rank > 0) then
            call MPI_recv(u_old(:, 0), num_points + 2, MPI_DOUBLE_PRECISION, &
                            rank - 1, 1, MPI_COMM_WORLD, status, error)
        end if

        ! Perform actual Jacobi
        du_max_proc = 0.d0
        do i=1, num_points
            do j=start_index, end_index
                u(i, j) = 0.25d0 * (u_old(i-1, j) + u_old(i+1, j) + u_old(i, j-1) + u_old(i, j+1) - dx**2 * f(i,j))
                du_max_proc = max(du_max_proc, abs(u(i,j) - u_old(i,j)))
            end do
        end do

        call MPI_allreduce(du_max_proc, du_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, error)

        if (rank == 0) then
            if (mod(N, PRINT_INTERVAL) == 0) then
                print '("After ",i8," iterations, du_max = ",d16.6,/)', N, du_max
            end if
        end if

        ! Check for exit tolerance
        if (du_max < tolerance) then
            exit
        end if
    end do

    ! Check for failure
    if (N >= MAX_ITERATIONS) then
        if (rank == 0) then
            print *, "*** Jacobi failed to converge!"
            print *, "***   Reached du_max = ", du_max
            print *, "***   Tolerance = ", tolerance
        end if
        call mpi_finalize(error)
        stop
    end if

    ! Output
    write(file_name, '("jacobi_",i1,".txt")') rank
    open(unit=IO_UNIT, file=file_name, status='replace')
    if (rank == 0) then
        write(IO_UNIT, *) u(:, 0)
    end if
    do j=start_index, end_index
        write(IO_UNIT, *) u(:, j)
    end do
    if (rank == num_procs - 1) then
        write(IO_UNIT, *) u(:, end_index + 1)
    end if
    close(IO_UNIT)

    ! Close out MPI
    call mpi_finalize(error)

end program jacobi_2d_mpi
