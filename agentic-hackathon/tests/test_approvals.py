from src.approvals import ApprovalRequest, AutoApproveProvider


def test_auto_approve_provider():
    provider = AutoApproveProvider()
    request = ApprovalRequest(task="task", intent="general", notes=["note"])
    decision = provider.request(request)
    assert decision.approved is True
