# 出处:landmark2face_wy/sync_batchnorm/replicate.pyc(py3.8,反汇编见
# evidence/modules/_heygem_orphans/disasm/modules_heygem_landmark2face_wy_sync_batchnorm_replicate.pyc.dis)。
# 该 .dis 模块 [Constants] 第一项是 0(不是字符串)=> 模块级没有 docstring,出处写在注释里。
# 恢复说明:三个函数/dp 类的 docstring 逐字取自 [Constants];函数体按 [Disassembly] 逐指令等价重建:
#   * patch_replication_callback 的 old_replicate 是闭包 cell 变量(STORE_DEREF),末尾是直接
#     `data_parallel.replicate = new_replicate`(字节码 LOAD_FAST new_replicate + STORE_ATTR,无 __get__);
#   * replicate 用显式 `super(DataParallelWithCallback, self).replicate(module, device_ids)`
#     (类体有 __classcell__,与字节码一致)。
# 未还原细节:无。
import functools

from torch.nn.parallel.data_parallel import DataParallel

__all__ = [
    'CallbackContext',
    'execute_replication_callbacks',
    'DataParallelWithCallback',
    'patch_replication_callback',
]


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """
    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate
