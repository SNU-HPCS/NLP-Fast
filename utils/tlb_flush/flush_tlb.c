/*
 * flush_tlb.c
 */
#include <linux/uaccess.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/percpu-defs.h>
#include <linux/smp.h>
#include <asm/tlbflush.h>
#include <asm/tlb.h>
#include <asm/mmu.h>
#include <asm/cpufeature.h>

/*#define CONFIG_PARAVIRT "y"*/

#ifdef CONFIG_PARAVIRT
#include <asm/paravirt.h>
#else
#define __flush_tlb() __native_flush_tlb()
#define __flush_tlb_global() __native_flush_tlb_global()
#define __flush_tlb_single(addr) __native_flush_tlb_single(addr)
#endif

MODULE_LICENSE("GPL");


/*static inline void __flush_tlb_all(void)*/
/*{*/
	/*if (cpu_has_pge)*/
		/*__flush_tlb_global();*/
	/*else*/
		/*__flush_tlb();*/
/*}*/

static void do_flush_tlb_all(void *info)
{
	/*count_vm_tlb_event(NR_TLB_REMOTE_FLUSH_RECEIVED);*/
	__flush_tlb_all();
	if (this_cpu_read(cpu_tlbstate.state) == TLBSTATE_LAZY)
		leave_mm(smp_processor_id());
}

static void custom_flush_tlb_all(void)
{
    /*count_vm_tlb_event(NR_TLB_REMOTE_FLUSH);*/
    on_each_cpu(do_flush_tlb_all, NULL, 1);
}


static long flush_tlb_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	int ret = 0;
	/*struct binder_proc *proc = filp->private_data;*/
	/*struct binder_thread *thread;*/
	unsigned int size = _IOC_SIZE(cmd);
	void __user *ubuf = (void __user *)arg;
    unsigned long addr;

    // copy an argument from user-level
    if (size != sizeof(unsigned long)) {
        ret = -EINVAL;
        goto out;
    }
    if (copy_from_user(&addr, ubuf, sizeof(addr))) {
        ret = -EFAULT;
        goto out;
    }

    if (addr) {
        // FIXME
        custom_flush_tlb_all();
    } else {
        custom_flush_tlb_all();
    }
out:
    return ret;
}

/*static int flush_tlb_open(struct inode *nodp, struct file *filp)*/
/*{*/
    /*return 0;*/
/*}*/

/*static int binder_release(struct inode *nodp, struct file *filp)*/
/*{*/
	/*return 0;*/
/*}*/

static const struct file_operations flush_tlb_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = flush_tlb_ioctl,
	.compat_ioctl = flush_tlb_ioctl,
	/*.open = flush_tlb_open,*/
	/*.release = binder_release,*/
};

static struct miscdevice binder_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "flush_tlb",
	.fops = &flush_tlb_fops
};

static int __init flush_tlb_init(void)
{
	int ret;

	ret = misc_register(&binder_miscdev);

	return ret;
}
module_init(flush_tlb_init)


static void __exit flush_tlb_exit(void)
{
    misc_deregister(&binder_miscdev);
}
module_exit(flush_tlb_exit)
