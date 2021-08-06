#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define FLUSH_TLB_CMD  _IOWR('b', 1, unsigned long)


int main()
{
    int ioctlfd;
    unsigned long addr;
    printf("Test program (TLB_FLUSHING)\n");

    ioctlfd = open("/dev/flush_tlb", O_RDWR);
    if (ioctlfd < 0) {
        perror("Open flush_tlb binder failed");
        return -1;
    }

    addr = 0;
    if (ioctl(ioctlfd, FLUSH_TLB_CMD, &addr)) {
        perror("why");
        return -1;
    }
}
