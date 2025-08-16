# run_sanity_check.py
from tickets.inference import predict_all

if __name__ == "__main__":
    subject = "Cannot access my account"
    body    = "Login fails with 2FA enabled after password reset."
    out = predict_all(subject, body)
    # For M2 we only verify structure; no UI, no ClickUp calls
    print(out)