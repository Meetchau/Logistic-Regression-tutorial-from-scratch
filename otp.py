import pyotp
 
totp = pyotp.TOTP("M25D2MBBSBIZ42QC")
print(totp.now())
 
 