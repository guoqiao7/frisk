// RUN: layout-reduce --round-trip %S/Inputs/layout-empty.json > %t.first
// RUN: layout-reduce --round-trip %t.first > %t.second
// RUN: diff %t.first %t.second
// RUN: layout-reduce --replay %t.second | FileCheck %s
// RUN: layout-reduce --reduce %t.second > %t.small
// RUN: layout-reduce --reduce %t.small > %t.again
// RUN: diff %t.small %t.again
// RUN: layout-reduce --replay %t.small | FileCheck %s
// RUN: if layout-reduce --round-trip %s > %t.bad 2>&1; then exit 1; fi
// RUN: FileCheck %s --check-prefix=BAD < %t.bad
// CHECK: reproduced EmptyDomain
// BAD: unsupported or malformed
