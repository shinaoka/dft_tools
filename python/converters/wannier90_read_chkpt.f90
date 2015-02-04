! -*- f90 -*-
MODULE read_chkpt
! ======================================================================
! based on subroutines from wannier90

!  * checks if selected keywords in seedname.win file are correct

!  * reads projectors from seedname.chkpt

!  * check if seedname.chk is conssistent with seedname.win
! ======================================================================

implicit none
 integer, parameter, public       :: dp = kind(1.0d0)
 real(kind=dp), parameter, public :: bohr = 0.529177210800000000
 real(kind=dp), parameter, public    :: eps6  = 1.0000000e-6
 real(kind=dp), parameter, public    :: eps5  = 1.0000000e-5

 character(len=50), public, save :: seedname
 integer, public, save           :: num_wann
 integer, public, save           :: num_bands
 integer, public, save           :: num_kpts
 integer, public, save           :: mp_grid(3)
 integer, public, save           :: nntot         ! total number of neighbours for each k-point
 integer, public, save           :: num_exclude_bands
 character(len=20), public, save :: checkpoint
 logical, public, save           :: disentanglement
 logical, public, save           :: have_disentangled
 real(kind=dp), public, save     :: omega_invariant

 integer, parameter, public      :: maxlen = 120  ! Max column width of input file
 

 complex(kind=dp), allocatable, save, public :: u_matrix_opt(:,:,:)
 complex(kind=dp), allocatable, save, public :: u_matrix(:,:,:)
 real(kind=dp), allocatable,    public, save ::kpt_cart(:,:) !kpoints in cartesians
 integer, public, save, allocatable :: ndimwin(:)
 integer, allocatable, public,save :: exclude_bands(:)
 logical, public, save, allocatable :: lwindow(:,:)
 real(kind=dp), allocatable,    public, save :: kpt_latt(:,:) !kpoints in lattice vecs


 real(kind=dp),     public, save :: real_metric(3,3)
 real(kind=dp),     public, save :: recip_metric(3,3)
 real(kind=dp),     public, save :: real_lattice(3,3)
 real(kind=dp),     public, save :: recip_lattice(3,3)
 real(kind=dp),     public, save :: cell_volume



 !private
 integer,private                            :: num_lines
 character(len=maxlen), allocatable,private :: in_data(:)
 logical,private                            :: ltmp

 !subroutines and functions
 public :: io_file_unit  !(taken from Wannier90)
 public :: io_error  !(taken from Wannier90 and modified)
 public :: param_in_file  !(taken from Wannier90)
 public :: param_get_keyword  !(taken from Wannier90)
 public :: param_get_keyword_block  !(taken from Wannier90)
 public :: param_get_keyword_vector  !(taken from Wannier90)
 public :: param_get_range_vector !(taken from Wannier90)
 public :: utility_recip_lattice  !(taken from Wannier90)
 public :: utility_metric  !(taken from Wannier90)
 public :: utility_lowercase  !(taken from Wannier90 and modified)

 public :: param_read !(taken from Wannier90 and modified)
 public :: param_read_chkpt  !(taken from Wannier90 and modified)
 public :: write_ndimwin
 public :: write_U_matrix
 public :: write_U_matrix_opt
 public :: set_seedname
 public :: get_have_disentangled
 public :: get_num_bands 
 public :: get_num_wann
 public :: get_num_kpts
 public :: summary
 public :: dealloc

contains

  !==================================================================!
  function io_file_unit()
  !==================================================================!
  !                                                                  !
  ! Returns an unused unit number                                    !
  ! (so we can open a file on that unit                              !
  !                                                                  !
  !===================================================================
  implicit none

  integer :: io_file_unit,unit
  logical :: file_open

  unit = 9
  file_open = .true.
  do while ( file_open )
     unit = unit + 1
     inquire( unit, OPENED = file_open )
  end do

  io_file_unit = unit


  return
  end function io_file_unit

  subroutine report (msg)

    character(len=*), intent(in) :: msg
    write(*, '(1x,a)') trim(msg)

    return
  end subroutine report

  subroutine io_error ( error_msg )
    !==================================================================!
    !                                                                  !
    ! Aborts giving error message  (to be called only on master node)  !
    !                                                                  !
    !===================================================================


         implicit none
         character(len=*), intent(in) :: error_msg

         write(*,*)  'Exiting.......'
         write(*, '(1x,a)') trim(error_msg)

         call exit(1)

         return
    end subroutine io_error

    subroutine param_read_chkpt()
    !=================================================!
    ! Read checkpoint file for                        !
    !=================================================!

    implicit none

    integer :: chk_unit,nkp,i,j,k,ntmp,ierr
    character(len=33) :: header
    real(kind=dp) :: tmp_latt(3,3), tmp_kpt_latt(3,num_kpts)
    integer :: tmp_excl_bands(1:num_exclude_bands),tmp_mp_grid(1:3)

    write(*,'(1x,3a)') 'Reading restart information from file ',trim(seedname),'.chk :'

    chk_unit=io_file_unit()
    open(unit=chk_unit,file=trim(seedname)//'.chk',status='old',form='unformatted',err=121)

    ! Read comment line
    read(chk_unit) header
    write(*,'(1x,a)',advance='no') trim(header)

    ! Consistency checks
    read(chk_unit) ntmp                           ! Number of bands
    if (ntmp.ne.num_bands) call io_error('param_read_chk: Mismatch in num_bands')
    read(chk_unit) ntmp                           ! Number of excluded bands
    if (ntmp.ne.num_exclude_bands) &
         call io_error('param_read_chk: Mismatch in num_exclude_bands')
    read(chk_unit) (tmp_excl_bands(i),i=1,num_exclude_bands) ! Excluded bands
    do i=1,num_exclude_bands
       if (tmp_excl_bands(i).ne.exclude_bands(i)) &
            call io_error('param_read_chk: Mismatch in exclude_bands')
    enddo
    read(chk_unit) ((tmp_latt(i,j),i=1,3),j=1,3)  ! Real lattice
    do j=1,3
       do i=1,3
          if (abs(tmp_latt(i,j)-real_lattice(i,j)).gt.eps6) &
               call io_error('param_read_chk: Mismatch in real_lattice')
       enddo
    enddo
    read(chk_unit) ((tmp_latt(i,j),i=1,3),j=1,3)  ! Reciprocal lattice
    do j=1,3
       do i=1,3
          if (abs(tmp_latt(i,j)-recip_lattice(i,j)).gt.eps6) &
               call io_error('param_read_chk: Mismatch in recip_lattice')
       enddo
    enddo
    read(chk_unit) ntmp                ! K-points
    if (ntmp.ne.num_kpts) &
         call io_error('param_read_chk: Mismatch in num_kpts')
    read(chk_unit) (tmp_mp_grid(i),i=1,3)         ! M-P grid
    do i=1,3
       if (tmp_mp_grid(i).ne.mp_grid(i)) &
            call io_error('param_read_chk: Mismatch in mp_grid')
    enddo
    read(chk_unit) ((tmp_kpt_latt(i,nkp),i=1,3),nkp=1,num_kpts)
    do nkp=1,num_kpts
       do i=1,3
          if (abs(tmp_kpt_latt(i,nkp)-kpt_latt(i,nkp)).gt.eps6) &
               call io_error('param_read_chk: Mismatch in kpt_latt')
       enddo
    enddo
    read(chk_unit) ntmp                ! nntot
    read(chk_unit) ntmp                ! num_wann
    if (ntmp.ne.num_wann) &
         call io_error('param_read_chk: Mismatch in num_wann')
    ! End of consistency checks

    read(chk_unit) checkpoint             ! checkpoint
    checkpoint=adjustl(trim(checkpoint))

    read(chk_unit) have_disentangled      ! whether a disentanglement has been performed

    if (have_disentangled) then

       read(chk_unit) omega_invariant     ! omega invariant

       ! lwindow
       if (.not.allocated(lwindow)) then
          allocate(lwindow(num_bands,num_kpts),stat=ierr)
          if (ierr/=0) call io_error('Error allocating lwindow in param_read_chkpt')
       endif
       read(chk_unit,err=122) ((lwindow(i,nkp),i=1,num_bands),nkp=1,num_kpts)

       ! ndimwin
       if (.not.allocated(ndimwin)) then
          allocate(ndimwin(num_kpts),stat=ierr)
          if (ierr/=0) call io_error('Error allocating ndimwin in param_read_chkpt')
       endif
       read(chk_unit,err=123) (ndimwin(nkp),nkp=1,num_kpts)

       ! U_matrix_opt
       if (.not.allocated(u_matrix_opt)) then
          allocate(u_matrix_opt(num_bands,num_wann,num_kpts),stat=ierr)
          if (ierr/=0) call io_error('Error allocating u_matrix_opt in param_read_chkpt')
       endif
       read(chk_unit,err=124) (((u_matrix_opt(i,j,nkp),i=1,num_bands),j=1,num_wann),nkp=1,num_kpts)

    endif

    ! U_matrix
    if (.not.allocated(u_matrix)) then
       allocate(u_matrix(num_wann,num_wann,num_kpts),stat=ierr)
       if (ierr/=0) call io_error('Error allocating u_matrix in param_read_chkpt')
    endif
    read(chk_unit,err=125) (((u_matrix(i,j,k),i=1,num_wann),j=1,num_wann),k=1,num_kpts)

    close(chk_unit)

    write(*,'(a/)') ' ... done'

    return

121 call io_error('Error opening '//trim(seedname)//'.chk in param_read_chkpt')
122 call io_error('Error reading lwindow from '//trim(seedname)//'.chk in param_read_chkpt')
123 call io_error('Error reading ndimwin from '//trim(seedname)//'.chk in param_read_chkpt')
124 call io_error('Error reading u_matrix_opt from '//trim(seedname)//'.chk in param_read_chkpt')
125 call io_error('Error reading u_matrix from '//trim(seedname)//'.chk in param_read_chkpt')

  end subroutine param_read_chkpt


    !===========================================================================!
  subroutine param_get_keyword(keyword,found,c_value,l_value,i_value,r_value)
    !===========================================================================!
    !                                                                           !
    !             Finds the value of the required keyword.                      !
    !                                                                           !
    !===========================================================================!

    implicit none

    character(*),      intent(in)  :: keyword
    logical          , intent(out) :: found
    character(*)     ,optional, intent(inout) :: c_value
    logical          ,optional, intent(inout) :: l_value
    integer          ,optional, intent(inout) :: i_value
    real(kind=dp)    ,optional, intent(inout) :: r_value

    integer           :: kl, in,loop,itmp
    character(len=maxlen) :: dummy

    kl=len_trim(keyword)

    found=.false.

    do loop=1,num_lines
       in=index(in_data(loop),trim(keyword))
       if (in==0 .or. in>1 ) cycle
       itmp=in+len(trim(keyword))
       if (in_data(loop)(itmp:itmp)/='=' &
            .and. in_data(loop)(itmp:itmp)/=':' &
            .and. in_data(loop)(itmp:itmp)/=' ') cycle
       if (found) then
          call io_error('Error: Found keyword '//trim(keyword)//' more than once in input file')
       endif
       found=.true.
       dummy=in_data(loop)(kl+1:)
       in_data(loop)(1:maxlen) = ' '
       dummy=adjustl(dummy)
       if( dummy(1:1)=='=' .or. dummy(1:1)==':') then
          dummy=dummy(2:)
          dummy=adjustl(dummy)
       end if
    end do

    if(found) then
       if( present(c_value) ) c_value=dummy
       if( present(l_value) ) then
          if (index(dummy,'t') > 0) then
             l_value=.true.
          elseif (index(dummy,'f') > 0) then
             l_value=.false.
          else
             call io_error('Error: Problem reading logical keyword '//trim(keyword))
          endif
       endif
       if( present(i_value) ) read(dummy,*,err=220,end=220) i_value
       if( present(r_value) ) read(dummy,*,err=220,end=220) r_value
    end if

    return

220 call io_error('Error: Problem reading keyword '//trim(keyword))


  end subroutine param_get_keyword

   !=======================================!
  subroutine param_in_file ( )
    !=======================================!
    ! Load the *.win file into a character  !
    ! array in_file, ignoring comments and  !
    ! blank lines and converting everything !
    ! to lowercase characters               !
    !=======================================!

    implicit none

    integer           :: in_unit,tot_num_lines,ierr,line_counter,loop,in1,in2
    character(len=maxlen) :: dummy
    integer           :: pos
    character, parameter :: TABCHAR = char(9)

    in_unit=io_file_unit( )
    open (in_unit, file=trim(seedname)//'.win',form='formatted',status='old',err=101)

    num_lines=0;tot_num_lines=0
    do
       read(in_unit, '(a)', iostat = ierr, err= 200, end =210 ) dummy
       ! [GP-begin, Apr13, 2012]: I convert all tabulation characters to spaces
       pos = index(dummy,TABCHAR)
       do while (pos .ne. 0)
          dummy(pos:pos) = ' '
          pos = index(dummy,TABCHAR)
       end do
       ! [GP-end]
       dummy=adjustl(dummy)
       tot_num_lines=tot_num_lines+1
       if( .not.dummy(1:1)=='!'  .and. .not. dummy(1:1)=='#' ) then
          if(len(trim(dummy)) > 0 ) num_lines=num_lines+1
       endif

    end do

101 call io_error('Error: Problem opening input file '//trim(seedname)//'.win')
200 call io_error('Error: Problem reading input file '//trim(seedname)//'.win')
210 continue
    rewind(in_unit)

    allocate(in_data(num_lines),stat=ierr)
    if (ierr/=0) call io_error('Error allocating in_data in param_in_file')

    line_counter=0
    do loop=1,tot_num_lines
       read(in_unit, '(a)', iostat = ierr, err= 200 ) dummy
       ! [GP-begin, Apr13, 2012]: I convert all tabulation characters to spaces
       pos = index(dummy,TABCHAR)
       do while (pos .ne. 0)
          dummy(pos:pos) = ' '
          pos = index(dummy,TABCHAR)
       end do
       ! [GP-end]
       call utility_lowercase(dummy)
       dummy=adjustl(dummy)
       if( dummy(1:1)=='!' .or.  dummy(1:1)=='#' ) cycle
       if(len(trim(dummy)) == 0 ) cycle
       line_counter=line_counter+1
       in1=index(dummy,'!')
       in2=index(dummy,'#')
       if(in1==0 .and. in2==0)  in_data(line_counter)=dummy
       if(in1==0 .and. in2>0 )  in_data(line_counter)=dummy(:in2-1)
       if(in2==0 .and. in1>0 )  in_data(line_counter)=dummy(:in1-1)
       if(in2>0 .and. in1>0 )   in_data(line_counter)=dummy(:min(in1,in2)-1)
    end do

    close(in_unit)

  end subroutine param_in_file


    !==================================================================!
  subroutine param_read ()
  !==================================================================!
  !                                                                  !
  ! Read parameters and calculate derived values                     !
  !                                                                  !
  !===================================================================

    implicit none

    !local variables
    real(kind=dp)  :: real_lattice_tmp(3,3)
    integer :: nkp,i_temp,ierr,iv_temp(3)
    logical :: found
    call param_in_file

    !%%%%%%%%%%%%%%%%
    !System variables
    !%%%%%%%%%%%%%%%%

    !num_wann
    num_wann      =   -99
    call param_get_keyword('num_wann',found,i_value=num_wann)
    if(.not. found) call io_error('Error: You must specify num_wann')
    if(num_wann<=0) call io_error('Error: num_wann must be greater than zero')

    !num_exclude_bands
    num_exclude_bands=0
    call param_get_range_vector('exclude_bands',found,num_exclude_bands,lcount=.true.)
    if(found) then
       if(num_exclude_bands<1) call io_error('Error: problem reading exclude_bands')
       allocate(exclude_bands(num_exclude_bands),stat=ierr)
       if (ierr/=0) call io_error('Error allocating exclude_bands in param_read')
       !exclude_bands
       call param_get_range_vector('exclude_bands',found,num_exclude_bands,.false.,exclude_bands)
       if (any(exclude_bands<1)  ) &
            call io_error('Error: exclude_bands must contain positive numbers')
    end if

    !num_bands
    call param_get_keyword('num_bands',found,i_value=i_temp)

    if(found) num_bands=i_temp
    if(.not.found) num_bands=num_wann
    if(found .and. num_bands<num_wann) then
        call io_error('Error: num_bands must be greater than or equal to num_wann')
    endif

    !mp_grid
    call param_get_keyword_vector('mp_grid',found,3,i_value=iv_temp)

    if(found) mp_grid=iv_temp
    if (.not. found) then
          call io_error('Error: You must specify dimensions of the Monkhorst-Pack grid by setting mp_grid')
    elseif (any(mp_grid<1)) then
          call io_error('Error: mp_grid must be greater than zero')
    endif

    !num_kpoints
    num_kpts= mp_grid(1)*mp_grid(2)*mp_grid(3)

    !disentanglement

    disentanglement=.false.
    if(num_bands>num_wann) disentanglement=.true.

    !real_lattice
    call param_get_keyword_block('unit_cell_cart',found,3,3,r_value=real_lattice_tmp)

    real_lattice=transpose(real_lattice_tmp)
    if(.not. found) call io_error('Error: Did not find the cell information in the input file')


    !recip_lattice
    call utility_recip_lattice(real_lattice,recip_lattice,cell_volume)
    call utility_metric(real_lattice,recip_lattice,real_metric,recip_metric)

    ! kpoints in cartesian coordinates
    allocate ( kpt_cart(3,num_kpts) ,stat=ierr)
    if (ierr/=0) call io_error('Error allocating kpt_cart in param_read')

    allocate ( kpt_latt(3,num_kpts) ,stat=ierr)
    if (ierr/=0) call io_error('Error allocating kpt_latt in param_read')

    call param_get_keyword_block('kpoints',found,num_kpts,3,r_value=kpt_cart)
    kpt_latt=kpt_cart
    if(.not. found) call io_error('Error: Did not find the kpoint information in the input file')

    ! Calculate the kpoints in cartesian coordinates
    do nkp=1,num_kpts
       kpt_cart(:,nkp)=matmul(kpt_latt(:,nkp),recip_lattice(:,:))
    end do

    deallocate(in_data,stat=ierr)
    if (ierr/=0) call io_error('Error deallocating in_data in param_read')

    return

  end subroutine param_read

  !===================================================================
  subroutine utility_recip_lattice (real_lat,recip_lat,volume)  !
    !==================================================================!
    !                                                                  !
    !  Calculates the reciprical lattice vectors and the cell volume   !
    !                                                                  !
    !===================================================================

    implicit none
    real(kind=dp) :: twopi=2*3.141592653589793238462643383279
    real(kind=dp), intent(in)  :: real_lat (3, 3)
    real(kind=dp), intent(out) :: recip_lat (3, 3)
    real(kind=dp), intent(out) :: volume

    recip_lat(1,1)=real_lat(2,2)*real_lat(3,3)-real_lat(3,2)*real_lat(2,3)
    recip_lat(1,2)=real_lat(2,3)*real_lat(3,1)-real_lat(3,3)*real_lat(2,1)
    recip_lat(1,3)=real_lat(2,1)*real_lat(3,2)-real_lat(3,1)*real_lat(2,2)
    recip_lat(2,1)=real_lat(3,2)*real_lat(1,3)-real_lat(1,2)*real_lat(3,3)
    recip_lat(2,2)=real_lat(3,3)*real_lat(1,1)-real_lat(1,3)*real_lat(3,1)
    recip_lat(2,3)=real_lat(3,1)*real_lat(1,2)-real_lat(1,1)*real_lat(3,2)
    recip_lat(3,1)=real_lat(1,2)*real_lat(2,3)-real_lat(2,2)*real_lat(1,3)
    recip_lat(3,2)=real_lat(1,3)*real_lat(2,1)-real_lat(2,3)*real_lat(1,1)
    recip_lat(3,3)=real_lat(1,1)*real_lat(2,2)-real_lat(2,1)*real_lat(1,2)

    volume=real_lat(1,1)*recip_lat(1,1) + &
         real_lat(1,2)*recip_lat(1,2) + &
         real_lat(1,3)*recip_lat(1,3)


    if( abs(volume) < eps5 ) then
       call io_error(' Found almost zero Volume in utility_recip_lattice')
    end if

    recip_lat=twopi*recip_lat/volume
    volume=abs(volume)

    return

  end subroutine utility_recip_lattice

   !=================================!

  subroutine utility_lowercase(string)
    !=================================!
    !                                 !
    ! Takes a string and converts to  !
    !      lowercase characters       !
    !                                 !
    !=================================!

    implicit none

    character(len=*), intent(inout) :: string
    character(len=maxlen) :: to_lowercase

    integer :: iA,iZ,idiff,ipos,ilett

    iA = ichar('A')
    iZ = ichar('Z')
    idiff = iZ-ichar('z')

    to_lowercase = string

    do ipos=1,len(string)
       ilett = ichar(string(ipos:ipos))
       if ((ilett.ge.iA).and.(ilett.le.iZ)) &
            to_lowercase(ipos:ipos)=char(ilett-idiff)
    enddo

    string = trim(adjustl(to_lowercase))

  end subroutine utility_lowercase

   !===================================================================
  subroutine utility_metric(real_lat,recip_lat, &
       real_metric,recip_metric)
    !==================================================================!
    !                                                                  !
    !  Calculate the real and reciprical space metrics                 !
    !                                                                  !
    !===================================================================
    implicit none

    real(kind=dp), intent(in)  :: real_lat(3,3)
    real(kind=dp), intent(in)  :: recip_lat(3,3)
    real(kind=dp), intent(out) :: real_metric(3,3)
    real(kind=dp), intent(out) :: recip_metric(3,3)

    integer :: i,j,l

    real_metric=0.0_dp ; recip_metric=0.0_dp

    do j=1,3
       do i=1,j
          do l=1,3
             real_metric(i,j)=real_metric(i,j)+real_lat(i,l)*real_lat(j,l)
             recip_metric(i,j)=recip_metric(i,j)+recip_lat(i,l)*recip_lat(j,l)
          enddo
          if(i.lt.j) then
             real_metric(j,i)=real_metric(i,j)
             recip_metric(j,i)=recip_metric(i,j)
          endif
       enddo
    enddo

  end subroutine utility_metric

  !==============================================================================================!
  subroutine param_get_keyword_block(keyword,found,rows,columns,c_value,l_value,i_value,r_value)
    !==============================================================================================!
    !                                                                                              !
    !                           Finds the values of the required data block                        !
    !                                                                                              !
    !==============================================================================================!

    implicit none

    character(*),      intent(in)  :: keyword
    logical          , intent(out) :: found
    integer,           intent(in)  :: rows
    integer,           intent(in)  :: columns
    character(*)     ,optional, intent(inout) :: c_value(columns,rows)
    logical          ,optional, intent(inout) :: l_value(columns,rows)
    integer          ,optional, intent(inout) :: i_value(columns,rows)
    real(kind=dp)    ,optional, intent(inout) :: r_value(columns,rows)

    integer           :: in,ins,ine,loop,i,line_e,line_s,counter,blen
    logical           :: found_e,found_s,lconvert
    character(len=maxlen) :: dummy,end_st,start_st

    found_s=.false.
    found_e=.false.

    start_st='begin '//trim(keyword)
    end_st='end '//trim(keyword)


    do loop=1,num_lines
       ins=index(in_data(loop),trim(keyword))
       if (ins==0 ) cycle
       in=index(in_data(loop),'begin')
       if (in==0 .or. in>1) cycle
       line_s=loop
       if (found_s) then
          call io_error('Error: Found '//trim(start_st)//' more than once in input file')
       endif
       found_s=.true.
    end do

    if(.not. found_s) then
       found=.false.
       return
    end if


    do loop=1,num_lines
       ine=index(in_data(loop),trim(keyword))
       if (ine==0 ) cycle
       in=index(in_data(loop),'end')
       if (in==0 .or. in>1) cycle
       line_e=loop
       if (found_e) then
          call io_error('Error: Found '//trim(end_st)//' more than once in input file')
       endif
       found_e=.true.
    end do

    if(.not. found_e) then
       call io_error('Error: Found '//trim(start_st)//' but no '//trim(end_st)//' in input file')
    end if

    if(line_e<=line_s) then
       call io_error('Error: '//trim(end_st)//' comes before '//trim(start_st)//' in input file')
    end if

    ! number of lines of data in block
    blen = line_e-line_s-1

    if ( (blen.ne.rows) .and. (blen.ne.rows+1) ) &
         call io_error('Error: Wrong number of lines in block '//trim(keyword))

    if ( (blen.eq.rows+1) .and. (index(trim(keyword),'unit_cell_cart').eq.0) ) &
         call io_error('Error: Wrong number of lines in block '//trim(keyword))


    found=.true.

    lconvert=.false.
    if (blen==rows+1) then
       dummy=in_data(line_s+1)
       if ( index(dummy,'ang').ne.0 ) then
          lconvert=.false.
       elseif ( index(dummy,'bohr').ne.0 ) then
          lconvert=.true.
       else
          call io_error('Error: Units in block '//trim(keyword)//' not recognised')
       endif
       in_data(line_s)(1:maxlen) = ' '
       line_s=line_s+1
    endif

!    r_value=1.0_dp
    counter=0
    do loop=line_s+1,line_e-1
       dummy=in_data(loop)
       counter=counter+1
       if( present(c_value) ) read(dummy,*,err=240,end=240) (c_value(i,counter),i=1,columns)
       if( present(l_value) ) then
          ! I don't think we need this. Maybe read into a dummy charater
          ! array and convert each element to logical
          call io_error('param_get_keyword_block unimplemented for logicals')
       endif
       if( present(i_value) ) read(dummy,*,err=240,end=240) (i_value(i,counter),i=1,columns)
       if( present(r_value) ) read(dummy,*,err=240,end=240) (r_value(i,counter),i=1,columns)
    end do

    if (lconvert) then
       if (present(r_value)) then
          r_value=r_value*bohr
       endif
    endif

    in_data(line_s:line_e)(1:maxlen) = ' '


    return

240 call io_error('Error: Problem reading block keyword '//trim(keyword))


  end subroutine param_get_keyword_block

   !=========================================================================================!
  subroutine param_get_keyword_vector(keyword,found,length,c_value,l_value,i_value,r_value)
    !=========================================================================================!
    !                                                                                         !
    !                  Finds the values of the required keyword vector                        !
    !                                                                                         !
    !=========================================================================================!

    implicit none

    character(*),      intent(in)  :: keyword
    logical          , intent(out) :: found
    integer,           intent(in)  :: length
    character(*)     ,optional, intent(inout) :: c_value(length)
    logical          ,optional, intent(inout) :: l_value(length)
    integer          ,optional, intent(inout) :: i_value(length)
    real(kind=dp)    ,optional, intent(inout) :: r_value(length)

    integer           :: kl, in,loop,i
    character(len=maxlen) :: dummy

    kl=len_trim(keyword)

    found=.false.



    do loop=1,num_lines
       in=index(in_data(loop),trim(keyword))
       if (in==0 .or. in>1 ) cycle
       if (found) then
          call io_error('Error: Found keyword '//trim(keyword)//' more than once in input file')
       endif
       found=.true.
       dummy=in_data(loop)(kl+1:)
       in_data(loop)(1:maxlen) = ' '
       dummy=adjustl(dummy)
       if( dummy(1:1)=='=' .or. dummy(1:1)==':') then
          dummy=dummy(2:)
          dummy=adjustl(dummy)
       end if
    end do

    if(found) then
       if( present(c_value) ) read(dummy,*,err=230,end=230) (c_value(i),i=1,length)
       if( present(l_value) ) then
          ! I don't think we need this. Maybe read into a dummy charater
          ! array and convert each element to logical
          call io_error('param_get_keyword_vector unimplemented for logicals')
       endif
       if( present(i_value) ) read(dummy,*,err=230,end=230) (i_value(i),i=1,length)
       if( present(r_value) ) read(dummy,*,err=230,end=230) (r_value(i),i=1,length)
    end if



    return

230 call io_error('Error: Problem reading keyword '//trim(keyword)//' in param_get_keyword_vector')


  end subroutine param_get_keyword_vector

   !====================================================================!
    subroutine param_get_range_vector(keyword,found,length,lcount,i_value)
    !====================================================================!
    !   Read a range vector eg. 1,2,3,4-10  or 1 3 400:100               !
    !   if(lcount) we return the number of states in length              !
    !====================================================================!

    implicit none

    character(*),      intent(in)    :: keyword
    logical          , intent(out)   :: found
    integer,           intent(inout) :: length
    logical,           intent(in)    :: lcount
    integer, optional, intent(out)   :: i_value(length)

    integer   :: kl, in,loop,num1,num2,i_punc
    integer   :: counter,i_digit,loop_r,range_size
    character(len=maxlen) :: dummy
    character(len=10), parameter :: c_digit="0123456789"
    character(len=2) , parameter :: c_range="-:"
    character(len=3) , parameter :: c_sep=" ,;"
    character(len=5) , parameter :: c_punc=" ,;-:"
    character(len=5)  :: c_num1,c_num2


    if(lcount .and. present(i_value) ) call io_error('param_get_range_vector: incorrect call')

    kl=len_trim(keyword)

    found=.false.

    do loop=1,num_lines
       in=index(in_data(loop),trim(keyword))
       if (in==0 .or. in>1 ) cycle
       if (found) then
          call io_error('Error: Found keyword '//trim(keyword)//' more than once in input file')
       endif
       found=.true.
       dummy=in_data(loop)(kl+1:)
       dummy=adjustl(dummy)
       if(.not. lcount) in_data(loop)(1:maxlen) = ' '
       if( dummy(1:1)=='=' .or. dummy(1:1)==':') then
          dummy=dummy(2:)
          dummy=adjustl(dummy)
       end if
    end do

    if(.not. found) return

    counter=0
    if (len_trim(dummy)==0) call io_error('Error: keyword '//trim(keyword)//' is blank')
    dummy=adjustl(dummy)
    do
       i_punc=scan(dummy,c_punc)
       if(i_punc==0) call io_error('Error parsing keyword '//trim(keyword))
       c_num1=dummy(1:i_punc-1)
       read(c_num1,*,err=101,end=101) num1
       dummy=adjustl(dummy(i_punc:))
       !look for range
       if(scan(dummy,c_range)==1) then
          i_digit=scan(dummy,c_digit)
          dummy=adjustl(dummy(i_digit:))
          i_punc=scan(dummy,c_punc)
          c_num2=dummy(1:i_punc-1)
          read(c_num2,*,err=101,end=101) num2
          dummy=adjustl(dummy(i_punc:))
          range_size=abs(num2-num1)+1
          do loop_r=1,range_size
             counter=counter+1
             if(.not. lcount) i_value(counter)=min(num1,num2)+loop_r-1
          end do
       else
          counter=counter+1
          if(.not. lcount) i_value(counter)=num1
       end if

       if(scan(dummy,c_sep)==1) dummy=adjustl(dummy(2:))
       if(scan(dummy,c_range)==1) call io_error('Error parsing keyword '//trim(keyword)//' incorrect range')
       if(index(dummy,' ')==1) exit
    end do

    if(lcount) length=counter
    if(.not.lcount) then
       do loop=1,counter-1
          do loop_r=loop+1,counter
             if(i_value(loop)==i_value(loop_r)) &
                call io_error('Error parsing keyword '//trim(keyword)//' duplicate values')
          end do
        end do
    end if

    return

101 call io_error('Error parsing keyword '//trim(keyword))


  end  subroutine param_get_range_vector
  

  function get_num_kpts()
     implicit none
     integer get_num_kpts
     get_num_kpts=num_kpts
  end function get_num_kpts


  function get_num_wann()
      implicit none
      integer get_num_wann
      get_num_wann=num_wann
  end function get_num_wann


  function get_num_bands()
      implicit none
      integer get_num_bands
      get_num_bands=num_bands
  end function get_num_bands


  function get_have_disentangled() 
    implicit none
    character(len=10)::  get_have_disentangled
    !to make sure
    !   True  -> 1 
    !   False -> 0 
    if (have_disentangled) then  
       get_have_disentangled =  "True"
    else
       get_have_disentangled =  "False"
    end if 
    return 	
  end function get_have_disentangled


  subroutine set_seedname(newname)
     implicit none
     character(len=50) ,      intent(inout)    :: newname
     seedname= newname
     return
  end subroutine set_seedname


  subroutine write_ndimwin()
      implicit none
      integer ::ndimwin_unit,nkp
      ndimwin_unit=io_file_unit()
      open(unit=ndimwin_unit,file='ndimwin_'//trim(seedname)//'.txt',action="write",&
      status='replace',form='formatted',position="rewind",err=121)

       do nkp=1,num_kpts
          write(ndimwin_unit,'(I0)') ndimwin(nkp)
       end do
    
      close(ndimwin_unit) 
      return

      121 call io_error('Error occured while writing to ndimwin_'//trim(seedname)//'.txt')
 
  end subroutine write_ndimwin
  

  subroutine write_U_matrix()
      implicit none
      integer ::U_matrix_unit,i,j,k
      U_matrix_unit=io_file_unit()
      open(unit=U_matrix_unit,file='U_matrix_'//trim(seedname)//'.txt',status='replace',&
      action="write",form='formatted',position="rewind",err=121)

      do k=1,num_kpts
         do j=1,num_wann
            do i=1,num_wann
               write(U_matrix_unit,'(2F25.17)') u_matrix(i,j,k)
            end do
         end do
      end do

      close(U_matrix_unit)
      return

      121 call io_error('Error occured while writing to U_matrix_'//trim(seedname)//'.txt')
      
  end subroutine write_U_matrix
  

  subroutine write_U_matrix_opt()
        implicit none
        integer ::U_matrix_opt_unit,i,j,nkp
        U_matrix_opt_unit=io_file_unit()
        open(unit=U_matrix_opt_unit,file='U_matrix_opt_'//trim(seedname)//'.txt',action="write",&
        status='replace',form='formatted',position="rewind",err=121)
	
        do nkp=1,num_kpts
          do j=1,num_wann
             do i=1,num_bands
                write(U_matrix_opt_unit,'(2F25.17)') u_matrix_opt(i,j,nkp)
             end do
          end do
        end do

        close(U_matrix_opt_unit)
        return

        121 call io_error('Error occured while writing to U_matrix_opt_'//trim(seedname)//'.txt')
        
  end subroutine write_U_matrix_opt

  subroutine summary()
    implicit none
    !prints out value of all important parameters
    character(len=maxlen) :: line
    character(len=30) :: fotmat_int="(A1,8X,A35,I3,20X,A1)"
    character(len=30) :: fotmat_grid_real="(A1,3X,3F20.16,3X,A1)"
    character(len=30) :: fotmat_grid_int="(A1,8X,A35,3I3,14X,A1)"
    character(len=30) :: fotmat_header="(A1,8X,A6,A32,20X,A1)"

    call report("!==================================================================!")
    call report("!                             Summary                              !")
    call report("!==================================================================!")

    write(line,fotmat_header) "!","file: ", trim(seedname)//".chk","!"
    call report(line)
    !num_wann 
    write(line,fotmat_int) "!","Number of Wannier orbitals is: ", num_wann,"!"
    call report(line)

    !num_bands
    write(line,fotmat_int) "!","Number of all bands: ", num_bands,"!"
    call report(line)
  
    !disentanglement
    if (have_disentangled) then
       call report("!        disentanglement had to be perfomed to obtain MLWF         !")
    else
       call report("!          No disentanglement was needed to obtain MLWF            !")
    endif   

    !num_exclude_bands
    if (have_disentangled) then
       write(line,fotmat_int) "!","Number of excluded bands is: ", num_exclude_bands,"!"
       call report(line)
    end if 

    !mp_grid
    write(line,fotmat_grid_int) "!","Monkhorst-Pack grid is: ", mp_grid(1),mp_grid(2),mp_grid(3),"!"
    call report(line)

    !num_kpts
    write(line,fotmat_int) "!","Number of k-points is: ",num_kpts,"!"
    call report(line)

    !real_lattice
    call report("!             real lattice is (in angstrom):                       !")
  
    write(line,fotmat_grid_real) "!",real_lattice(1,1),real_lattice(1,2),real_lattice(1,3),"!" 
    call report(line)
    write(line,fotmat_grid_real) "!",real_lattice(2,1),real_lattice(2,2),real_lattice(2,3),"!" 
    call report(line)
    write(line,fotmat_grid_real) "!",real_lattice(3,1),real_lattice(3,2),real_lattice(3,3),"!" 
    call report(line)

    !recip_lattice
    call report("!             reciprocal lattice is (in 1/angstrom):               !")

    write(line,fotmat_grid_real) "!",recip_lattice(1,1),recip_lattice(1,2),recip_lattice(1,3),"!" 
    call report(line) 
    write(line,fotmat_grid_real) "!",recip_lattice(2,1),recip_lattice(2,2),recip_lattice(2,3),"!" 
    call report(line) 
    write(line,fotmat_grid_real) "!",recip_lattice(3,1),recip_lattice(3,2),recip_lattice(3,3),"!" 
    call report(line) 

    call report("!==================================================================!")
    call report("!                         End of Summary                           !")
    call report("!==================================================================!")

  end subroutine summary 

  subroutine dealloc()
    implicit none
    integer :: ierr
    if ( allocated(kpt_latt) ) then
       deallocate ( kpt_latt, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating kpt_latt')
    endif

    if ( allocated(kpt_cart) ) then
       deallocate ( kpt_cart, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating kpt_cart')
    endif

    if( allocated( exclude_bands ) ) then
       deallocate( exclude_bands, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating exclude_bands')
    end if

    if( allocated( in_data ) ) then
       deallocate( in_data, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating in_data')
    end if

    if( allocated(u_matrix ) ) then
       deallocate( u_matrix, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating u_matrix')
    end if

    if( allocated(u_matrix_opt ) ) then
       deallocate( u_matrix_opt, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating u_matrix_opt')
    end if

    if( allocated(ndimwin ) ) then
       deallocate( ndimwin, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating ndimwin')
    end if

    if( allocated(lwindow ) ) then
       deallocate( lwindow, stat=ierr  )
       if (ierr/=0) call io_error('Error in deallocating lwindow')
    end if
    return
  end subroutine dealloc

END MODULE  read_chkpt
