#include "cml/cml.h"
#include <cblas.h>

void init(void)
{
    const double values[1] = {0.0};
    (void)cblas_ddot(1, values, 1, values, 1);
}
