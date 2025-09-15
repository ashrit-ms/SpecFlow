@echo off
echo ========================================
echo SpecECD Firewall Configuration
echo ========================================
echo.
echo This script will configure Windows Firewall to allow
echo SpecECD cloud server connections on port 8765
echo.
echo You need to run this as Administrator!
echo.
pause

echo Configuring Windows Firewall...
echo.

REM Add firewall rule for inbound connections
netsh advfirewall firewall add rule name="SpecECD Cloud Server" dir=in action=allow protocol=TCP localport=8765

if %errorlevel% equ 0 (
    echo ✓ Firewall rule added successfully
    echo   Rule name: SpecECD Cloud Server
    echo   Direction: Inbound
    echo   Protocol: TCP
    echo   Port: 8765
) else (
    echo ✗ Failed to add firewall rule
    echo   Make sure you're running as Administrator
    goto :error
)

echo.
echo ========================================
echo Firewall Configuration Complete
echo ========================================
echo.
echo The cloud server should now be accessible from other machines
echo on your local network.
echo.
echo To remove this rule later, run:
echo   netsh advfirewall firewall delete rule name="SpecECD Cloud Server"
echo.
goto :end

:error
echo.
echo ========================================
echo Manual Firewall Configuration
echo ========================================
echo.
echo Please manually configure Windows Firewall:
echo 1. Open Windows Defender Firewall
echo 2. Click "Advanced settings"
echo 3. Select "Inbound Rules"
echo 4. Click "New Rule..."
echo 5. Select "Port" and click Next
echo 6. Select "TCP" and enter port 8765
echo 7. Select "Allow the connection"
echo 8. Apply to all profiles
echo 9. Name it "SpecECD Cloud Server"
echo.

:end
pause
