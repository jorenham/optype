import optype as opt


def test_iterable():
    assert opt.inspect.is_iterable([])
    assert opt.inspect.is_iterable(())
    assert opt.inspect.is_iterable('')
    assert opt.inspect.is_iterable(b'')
    assert opt.inspect.is_iterable(range(2))
    assert opt.inspect.is_iterable(i for i in range(2))


# TODO: tests for is_runtime_protocol
# TODO: tests for get_args
# TODO: tests for get_protocol_members
# TODO: tests for get_protocols
