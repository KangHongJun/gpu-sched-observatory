from gso.bench.workloads import select_gpu


def test_select_gpu_pins_pci_order_and_device():
    env: dict[str, str] = {}
    select_gpu(1, env)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_select_gpu_overwrites_existing_order():
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0"}
    select_gpu(1, env)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
