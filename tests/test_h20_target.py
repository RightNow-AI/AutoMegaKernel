from schedule.ir import TARGETS


def test_h20_target_profile():
    target = TARGETS["h20"]

    assert target.name == "h20"
    assert target.sm_arch == 90
    assert target.num_sms == 78

    assert target.smem_bytes_per_sm == 233472
    assert target.smem_bytes_per_block_optin == 232448

    assert target.regs_per_sm == 65536
    assert target.max_threads_per_sm == 2048
    assert target.max_regs_per_thread == 255

    assert target.l2_bytes == 60 * 1024 * 1024
    assert target.hbm_bytes == 96 * 1024**3

    assert target.hbm_bandwidth_gbs == 4000.0
    assert target.measured_bw_gbs == 3712.3

    assert target.supports_cooperative is True
    assert target.wddm_tdr is False
