      PROGRAM StoG_new

!      WS Howells   June 1989
!      Matt Tucker(MGT) Sept 2009 (a 20 year slight upgrade)
      implicit none

      double precision :: ytemp

      integer, parameter :: mn=33000, maxqpts=20000
      double precision, allocatable :: xin(:),yin(:),xout(:),yout(:),y(:)
      double precision, allocatable :: ftxin(:),ftyin(:),ftxout(:),ftyout(:)
      double precision, allocatable :: xin_m(:,:),yin_m(:,:),xout_m(:),yout_m(:),yweight_m(:) 
      double precision :: delr,rmax,vmult,vadd,xd,pi,a,rho,delq,v,pi4r, &
                          yDS,SINUS1,SINUS,f1,f2,fs,afact,rp,vm,vp,r,average
      double precision :: xnew(mn),yw(mn),xoffset
      double precision :: maxx,minx,qmin,qmax,xdiv,yscale,yoffset,fscale,ftcut
      double precision :: rmccut,rmcmin,rmcmax
      integer :: lptin,lptout,i,n,nn,nr,ftpts,ftptsout,ftoffset
      integer :: nfiles,loop,loop2,noutpts,inpts,mainloop,test
      integer, allocatable :: npts(:)
      logical :: lyes, runft

      CHARACTER(len=10) :: char,yes
      character(len=32) :: version='new 3.1, oct 2011'
      character(len=80) :: filein,fileout,sqfileout,rmcsqfile,rmcgrfile,ftsqfile,ftgrfile,lorgrfile
      character(len=80), allocatable :: file(:),title(:) 
      
      LOGICAL :: LMOD

!      DANGLE=38.1*1.112/150.0/150.0
      PI=dacos(-1.0d0)

!      call TRANSFORM_IN
!      if(lptin.eq.0)then
!        STOP 'stog> ERROR ** No data'
!      endif
!

!
!       MGT: Put in part to merge a number of files. 
!            This assumes the each data set are binned on the same scale

          write(*,*) 'This is stog version',version
!
        write(*,'(a,$)') 'Enter number of files to include: '
        read(*,*) nfiles
        allocate (file(nfiles))
        allocate (npts(nfiles))
        allocate (title(nfiles))
        allocate (xin_m(nfiles,maxqpts))
        allocate (yin_m(nfiles,maxqpts))
        

        do loop=1,nfiles


        write(*,'(a,$)') 'Enter file name: '
        read(*,'(a)') file(loop)
        write(*,'(a,$)') 'enter qmin and qmax: ' 
        read(*,*) qmin,qmax
        write(*,'(a,$)') 'enter yoffset and yscale: '
        read(*,*) yoffset,yscale
        write(*,'(a,$)') 'enter Q offset: '
        read(*,*) xoffset

        open(10,file=file(loop))

        read(10,*) npts(loop)
        if (npts(loop)>maxqpts) then
          write(*,*) 'Too many Q points',loop,npts(loop)
          stop
        endif
        read(10,'(a)') title(loop)
       
        inpts=1
        do loop2=1,npts(loop)

        read(10,*) xin_m(loop,inpts),yin_m(loop,inpts)
        xin_m(loop,inpts)=xin_m(loop,inpts)+xoffset
        if (xin_m(loop,inpts).ge.qmin.and.xin_m(loop,inpts).lt.qmax) then
        yin_m(loop,inpts)=(yin_m(loop,inpts)/yscale)+yoffset
        inpts=inpts+1
        endif
        
        enddo
        npts(loop)=inpts-1
        close(10)
        enddo


        minx=xin_m(1,1)
        maxx=xin_m(1,1)

        do loop=1,nfiles
          do loop2=1,npts(loop)
          minx=min(minx,xin_m(loop,loop2))
          maxx=max(maxx,xin_m(loop,loop2))
          enddo
        enddo
        

        xdiv=xin_m(1,2)-xin_m(1,1)
        xdiv=dble(nint(xdiv*10000))/10000.0
        write(*,*) 'X div',xdiv

        noutpts=int((maxx-minx)/xdiv)
        write(*,*) 'number of points in combined spectrum',noutpts
        allocate(xout_m(noutpts))
        allocate(yout_m(noutpts))
        allocate(yweight_m(noutpts))
        
       minx=minx-xdiv
       do mainloop=1,noutpts
          xout_m(mainloop)=minx+dble(mainloop)*xdiv
          yout_m(mainloop)=0.0d0
          yweight_m(mainloop)=0.0d0
        do loop=1,nfiles
            do loop2=1,npts(loop)
         if (nint((xout_m(mainloop)/xdiv)-nint(xin_m(loop,loop2)/xdiv)).eq.0) then
!           if (loop.eq.3) write(*,*) xout_m(mainloop),xin_m(loop,loop2)
             yout_m(mainloop)=yout_m(mainloop)+yin_m(loop,loop2)
             yweight_m(mainloop)=yweight_m(mainloop)+1
!         else
!           test=nint((xout_m(mainloop)/xdiv)-nint(xin_m(loop,loop2)/xdiv))
!           if (loop.eq.nfiles.and.abs(test).lt.2) then
!           write(*,*) 'There is something wrong with the binning'
!           write(*,*) xout_m(mainloop),xin_m(loop,loop2)
!           write(*,*) test
!               endif
         endif
            enddo
        enddo
       enddo

       allocate(xin(noutpts),yin(noutpts),y(noutpts))


	  do loop=1,noutpts
	     xin(loop)=xout_m(loop)
	     yin(loop)=yout_m(loop)
         if (yweight_m(loop).gt.0) then            
            yin(loop)=yin(loop)/yweight_m(loop)
            if (nint(yweight_m(loop)) > nfiles) write(*,*) 'yweight', yweight_m(loop)
         else
            write(*,*) 'There is no data in this region', xin(loop)
         endif
        enddo
        
       lptin=noutpts
       
       write(*,'(a,$)') 'Enter file name for merged S(Q): '
        read(*,'(a)') sqfileout
        
!        open(10,file=sqfileout)
!        write(10,*) noutpts
!        write(10,'(a)') title(1) 
!        do loop=1,noutpts
!        write(10,*) xin(loop),yin(loop),yweight_m(loop)
!        enddo
!        close(10)


!      write(6,'(a,$)') 'Input filename: '
!      read(5,'(a)') filein
!      open(11,file=filein,status='old')
!      read(11,*) lptin
!      allocate(xin(lptin),yin(lptin),y(lptin))
!      read(11,'(a)') in_title
!      do i=1,lptin
!         read(11,*) xin(i),yin(i)
!      enddo
!      close(11)


!
!     MGT : Carry on as before 
!

      write(6,'(a,$)') 'Output filename for G(r): '
      read(5,'(a)') fileout
!
      write(6,1007)
      read(5,*) RMAX
      write(6,1006)
      read(5,*) lptout
      allocate(xout(lptout),yout(lptout))
         if(lptout.GT.mn)then
        write(6,1009)lptout
        lptout=mn
       endif
      delr=RMAX/lptout
      write(6,1005)delr

      write(6,1001)
      read(5,*) char
!      if(kr.eq.0)then
!       LMOD=.false.
!      else
       if(char(1:1).eq.'Y'.OR.char(1:1).eq.'y')then
        LMOD=.true.
       else
        LMOD=.false.
       endif
!        if (LMOD) then
!       write(*,'(a,$)') 'Enter file name for lorched FT corrected G(r): '
!        read(*,'(a)') lorgrfile
!       endif
!      endif
      write(6,1008)
      read(5,*) RHO
    
    lyes = .true.
    do while (lyes)
    
   
    
      write(6,'(a,$)') 'Add values: '
      read(5,*) vadd
      vmult = 1.0d0/(1.0d0+vadd)    
      write(6,'(a)') '(F(Q) must tend to 1.0 at high Q)'
      
         open(10,file=sqfileout)
        write(10,*) noutpts
        write(10,'(a)') title(1) 
!        do loop=1,noutpts
!        write(10,*) xin(loop),yin(loop),yweight_m(loop)
!        enddo
!        close(10)

      do i=1,lptin
         y(i)=yin(i)/vmult+vadd
         write(10,*) xin(i),y(i)
      enddo
      close(10)
      maxx=xin(lptin)
    
     call stog_bit(lptin,lptout,xin,y,delr,RHO,.false.,xout,yout)   
!      call testsub(lptin,lptout,xin,y,xout,yout)
     
     
     
!      xcaptout='    R (Angstrom)     '
!      ycaptout=' Radial Distribution    G of R'
!
      open(11,file=fileout,status='unknown')
      write(11,*) lptout
!      write(11,'(a)') in_title
      write(11,'(a)') 'r , G(r) , D(r) , T(r)'
      do i=1,lptout
!         write(11,*) xout(i),',',yout(i),' , ',xout(i)*(yout(i)-1.0d0),' , ', &
 !                    xout(i)*yout(i)
          write(11,*) xout(i),yout(i)
      enddo
      close(11)
      
      average = 0.0d0
      i = 1
      do while (xout(i)<=1.01d0)
        average = average + yout(i)**2
        i = i + 1
      end do
      write(6,'(a,g10.4)') 'Low-r mean-square value: ',dsqrt(average)
      write(6,'(a,$)') 'Try again [Y/N]: '
      read(5,'(a)') yes
      yes = adjustl(yes)
      lyes = (yes(1:1)=='Y')
    end do
    
      runft=.false.
      write(6,'(a,$)') 'Fourier filter data [Y/N]: '
      read(5,'(a)') yes
      yes = adjustl(yes)
      runft = (yes(1:1)=='Y'.or.yes(1:1)=='y')
    
           
    ft:if (runft) then
       allocate(ftxin(lptout),ftyin(lptout))
       
       write(6,'(a,$)') 'Please enter r cut-off: '
       read(5,*) ftcut
       write(*,*) 'FT cut-off set to ',ftcut
       write(*,'(a,$)') 'Enter file name for FT corrected S(Q): '
        read(*,'(a)') ftsqfile
       write(*,'(a,$)') 'Enter file name for FT corrected G(r): '
        read(*,'(a)') ftgrfile
       
 
       ftpts=0
        print *, 'DEBUG LINES STILL HERE'
       do loop=1,lptout
       if (xout(loop) <= ftcut) then
       ftpts=ftpts+1
       ftxin(ftpts)=xout(loop)
       ftyin(ftpts)=yout(loop)+1
!       print *, loop, xout(loop), ftxin(ftpts), yout(loop), ftyin(ftpts)
       endif
       enddo

       ftptsout=nint(maxx/xdiv)

       allocate(ftxout(ftptsout),ftyout(ftptsout))
       write(*,*) 'Number of ft points is', ftptsout
       call stog_bit(ftpts,ftptsout,ftxin,ftyin,xdiv,RHO,.false.,ftxout,ftyout)
       
       open(10,file='ft.dat')
       ftoffset=1
       write(10,*) ftptsout
       write(10,*) 'The FT correction function'
        print *, 'DEBUG LINES STILL HERE'
       do loop=1,ftptsout
        ytemp = ftyout(loop)
       ftyout(loop)=((ftyout(loop)-1)*(rho*rho*(2*pi)**3))+1
        print *, ftxout(loop), ytemp, ftyout(loop)
       write(10,*) ftxout(loop),ftyout(loop)
       if (nint(ftxout(loop)/xdiv) == nint(xin(1)/xdiv)) ftoffset=loop
       enddo
       close(10)
       if (ftoffset.eq.1) then 
       write(*,*) ftxout(loop),xin(1)
       stop 'FT binning problem'
       endif
       ftoffset=ftoffset-1
       
        stop

       write(*,*) 'FT offset', ftoffset
    
       do loop=1,lptin
!       if (nint((ftxout(loop+ftoffset)-(xdiv/2))/xdiv) == nint((xin(loop)-(xdiv/2))/xdiv)) then
       if (nint(ftxout(loop+ftoffset)/xdiv) == nint(xin(loop)/xdiv)) then       
       y(loop)=(y(loop)-(ftyout(loop+ftoffset)))+1
       else 
       write(*,*) nint(ftxout(loop+ftoffset)/xdiv),nint(xin(loop)/xdiv)
!        write(*,*) nint((ftxout(loop+ftoffset)-(xdiv/2))/xdiv),nint((xin(loop)-(xdiv/2))/xdiv)

       stop 'FT correction binning mismatch'
       end if
       enddo
       
       open(10,file=ftsqfile)
       write(10,*) lptin
       write(10,'(a,a)') 'Ft corrected sq', title(1)
       do loop=1,lptin
       write(10,*) xin(loop),y(loop)
       enddo
       close(10)
       
       call stog_bit(lptin,lptout,xin,y,delr,RHO,.false.,xout,yout)
       
       open(10,file=ftgrfile)
       write(10,*) lptout
       write(10,'(a,a)') 'Ft corrected gr and dr', title
       do loop=1,lptout
!       write(10,*) xout(loop),yout(loop),yout_e(loop)
        write(10,*) xout(loop),yout(loop),(yout(loop)-1)*xout(loop)
       enddo
       close(10)

       endif  ft


       if (LMOD) then 
       
       write(*,'(a,$)') 'Enter file name for lorched FT corrected G(r): '
        read(*,'(a)') lorgrfile
    
       call stog_bit(lptin,lptout,xin,y,delr,RHO,LMOD,xout,yout)
       
       open(10,file=lorgrfile)
       write(10,*) lptout
       write(10,'(a,a)') 'Ft corrected gr', title(1)
       do loop=1,lptout
       write(10,*) xout(loop),yout(loop)
       enddo
       close(10)
      
       endif 
    
       write(6,'(a,$)') 'Please enter the final scale number: ' 
       read(*,*) fscale
       write(6,'(a,$)') 'Filename for RMC S(Q): '
       read(*,'(a)') rmcsqfile
       write(6,'(a,$)') 'Filename for RMC G(r): ' 
       read(*,'(a)') rmcgrfile
       write(*,'(a,$)') 'RMC cuttoff, 1st peak min and max'
       read(*,*) rmccut,rmcmin,rmcmax
       open(10,file=rmcsqfile)
       open(11,file=rmcgrfile)
       
        write(10,*) noutpts
        write(10,'(a,a)') 'rmc S(Q)',title(1) 
        do loop=1,noutpts
        write(10,*) xin(loop),(y(loop)-1)*fscale
        enddo
        close(10)

        write(11,*) lptout
        write(11,'(a,a)') 'rmc G(r)',title(1) 
        do loop=1,lptout
        if ((xout(loop).le.rmccut).and.(xout(loop).ge.rmcmax.or.xout(loop).le.rmcmin)) yout(loop)=0
        write(11,*) xout(loop),(yout(loop)-1)*fscale
        enddo
        close(11)
       
        
!
!      call TRANSFORM_OUT
      STOP 'stog> ouput is in point mode'
 1001   FORMAT(' stog> window function (Y/N) [N] ? ',$)
 1005     FORMAT(' stog> increment in r =',f6.3)
 1006     FORMAT(' stog> number of r points ? ',$)
 1007     FORMAT(' stog> max r value ? ',$)
 1008     FORMAT(' stog> number density (atoms/A3) ? ',$)
 1009     FORMAT(' stog> #points reduced from ',I5,' to ',I5)
! 1010     FORMAT(Q,F10.0)
! 1011     FORMAT(Q,I)
! 1012     FORMAT(q,a)

      END program stog_new

      subroutine stog_bit(lptin,lptout,xin,y,delr,RHO,LMOD,xout,yout)

      implicit none

      integer, parameter :: mn=33000, maxqpts=20000
      double precision   xin(*),xout(*),yout(*),y(*) 
      double precision :: delr,rmax,vmult,vadd,xd,pi,a,rho,delq,v,pi4r, &
                          yDS,SINUS1,SINUS,f1,f2,fs,afact,rp,vm,vp,r,average
      double precision :: xnew(mn),yw(mn), ynew(mn)
      integer :: lptin,lptout,i,n,nn,nr
  
      LOGICAL :: LMOD

      write(*,*) 'In stog bit'
!      DANGLE=38.1*1.112/150.0/150.0
      PI=dacos(-1.0d0)


!      IF (.not.ALLOCATED(xin)) ALLOCATE(xin(lptin))
!      IF (.not.ALLOCATED(y)) ALLOCATE(y(lptin))
!      IF (.not.ALLOCATED(xout)) ALLOCATE(xout(lptout))
!      IF (.not.ALLOCATED(yout)) ALLOCATE(yout(lptout))
      
!      allocate(xin(lptin),y(lptin))
!      allocate(xout(lptout),yout(lptout))

       if(LMOD)then
          A=PI/xin(lptin)
        do nn=1,lptin
          yw(nn)=SIN(xin(nn)*A)/xin(nn)/A
        end do
       else
        do nn=1,lptin
          yw(nn)=1.0
        end do
       endif



!      do i=1,lptin
!         y(i)=yin(i)
!      enddo
    
!      user_par_out(15)=RHO
!
!  FORM VECTOR OF EQUALLY-SPACED R'S AT WHICH THE  FOURIER TRANSFORM
!  IS TO BE COMPUTED (RMAX IN ANGSTROMS), AND THE NUMBER OF R-POINTS.
!
        
        DO NR=1,lptout
   
       xout(NR)=delr*dble(NR)
      
      end do
      
         
!
!  THE NUMBER OF POINTS IN THE RANGE OF DATA TO BE TRANSFORMED.
!
                                             !half x-channel
      xd=(xin(2)-xin(1))/2.0d0
      IF (LMOD) then
      open(10,file="ft_sq.dat")
      write(10,*) lptin
      write(10,*) 'The function transformed'
      ENDIF
      do n=1,lptin
!       xnew(n)=xin(n) +xd                  !offset - mid channel
                                       !not removing a Q hist offset
       xnew(n)=xin(n)
                                       !Qi(Q)
       ynew(n)=yw(n)*(y(n)-1.0d0)*xnew(n)
       IF (LMOD) write(10,*) xnew(n),ynew(n)/xnew(n)+1 
            
       
      end do
      IF (LMOD) close(10)

 

!
!  COMPUTE FOURIER TRANSFORM OF THE DATA
!
      delq=(xin(lptin)-xin(1))/(lptin-1)
        AFACT=delq*2.0/PI
        DO NR=1,lptout
        FS=0.0d0
        RP=xout(NR)
         DO N=2,lptin
          SINUS1=SIN(xnew(N-1)*RP)
          SINUS=SIN(xnew(N)*RP)
        FS=FS+ (SINUS*ynew(N)+SINUS1*ynew(N-1))/2.0d0
!        print *, N, xnew(N), SINUS, SINUS1, yw(N), yw(N-1), xnew(N)*(y(N)-1.d0), xnew(N-1)*(y(N-1)-1.d0), FS
       end do
        yout(NR)=FS*AFACT
      END DO
!
!               COMPUTE CORRECTION TO THE TRANSFORM
!               FOR THE OMITTED SMALL-Q REGION
!
        DO NR=1,lptout
        R=xout(NR)
        V=xnew(1)*R
         if(LMOD)then
          VM=xnew(1)*(R-A)
          VP=xnew(1)*(R+A)
          F1=((VM*SIN(VM)+COS(VM)-1.0)/(R-A)**2-(VP*SIN(VP)+COS(VP)-1.0)/(R+A)**2)/2.0/A
          F2=(SIN(VM)/(R-A)-SIN(VP)/(R+A))/2.0/A
       else
          F1=(2.*V*SIN(V)-(V*V-2.)*COS(V)-2.)/R/R/R
          F2=(SIN(V)-V*COS(V))/R/R
       endif
        yDS=(2./PI)*(F1*y(1)/xin(1)-F2)
                                   !D(R)-rho
        yout(NR)=yout(NR)+yDS
      END DO

      pi4r=PI*4.0d0*RHO
      do n=1,lptout
       yout(n)=yout(n)/pi4r/xout(n) +1.0d0
      end do
      return

end subroutine stog_bit

      subroutine testsub(lptin,lptout,xin,y,xout,yout)
      
       implicit none

      double precision, allocatable ::    xin(:),y(:),xout(:),yout(:)
      integer :: lptin,lptout,i,n,nn,nr
      
      write(*,*) 'xin', ALLOCATED(xin)
      write(*,*) 'xin', ALLOCATED(xout) 
      write(*,*) 'xin', ALLOCATED(y)
      write(*,*) 'xin', ALLOCATED(yout) 
      
      allocate(xin(lptin),y(lptin))
      allocate(xout(lptout),yout(lptout))
      
      write(*,*) 'xin', ALLOCATED(xin)
      write(*,*) ALLOCATED(xout) 
      
      do nr=1,lptin
      xout(nr)=dble(nr)
      enddo
      
      return
      
      end subroutine testsub 
