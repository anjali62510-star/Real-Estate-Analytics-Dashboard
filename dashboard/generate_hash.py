import bcrypt

password = "admin123"

hashed = bcrypt.hashpw(
    password.encode(),
    bcrypt.gensalt(rounds=12)
).decode()

print(hashed)