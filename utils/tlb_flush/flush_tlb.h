#ifndef FLUSH_TLB_H
#define FLUSH_TLB_H

struct flush_tlb_parv_t {
    // [addr == 0] => flush all
    // [addr != 0] => flush an entry containing the addr
    unsigned long addr;
};

#endif
