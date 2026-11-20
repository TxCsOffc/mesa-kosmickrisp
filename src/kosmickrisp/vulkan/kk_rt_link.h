#ifndef KK_RT_LINK_H
#define KK_RT_LINK_H 1

#include "kosmickrisp/bridge/mtl_rt_function.h"

#include "util/u_atomic.h"

#include <stdint.h>
#include <stdlib.h>

struct kk_rt_link {
   int32_t ref;
   uint32_t count;
   uint32_t table_size;
   mtl_binary_function **functions;
   uint32_t *indices;
};

static inline struct kk_rt_link *
kk_rt_link_ref(struct kk_rt_link *link)
{
   if (link)
      p_atomic_inc(&link->ref);
   return link;
}

static inline void
kk_rt_link_unref(struct kk_rt_link *link)
{
   if (link && p_atomic_dec_zero(&link->ref)) {
      free(link->functions);
      free(link->indices);
      free(link);
   }
}

void kk_rt_set_pending_link(struct kk_rt_link *link);
struct kk_rt_link *kk_rt_get_pending_link(void);

#endif
