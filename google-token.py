from google_auth_oauthlib.flow import InstalledAppFlow

# Установите область доступа к Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

flow = InstalledAppFlow.from_client_secrets_file('./credential.json', SCOPES)
creds = flow.run_local_server(port=0)

# Печать refresh token
print('Refresh token:', creds.refresh_token)
