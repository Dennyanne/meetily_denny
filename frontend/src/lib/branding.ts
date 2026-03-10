const getString = (value: string | undefined, fallback: string) => {
  const trimmed = value?.trim();
  return trimmed ? trimmed : fallback;
};

const getBoolean = (value: string | undefined, fallback: boolean) => {
  if (value === undefined) {
    return fallback;
  }

  return value.trim().toLowerCase() === 'true';
};

export const LEGACY_APP_NAME = 'Meetily';
export const APP_NAME = getString(process.env.NEXT_PUBLIC_APP_NAME, '회의관리시스템');
export const APP_DESCRIPTION = getString(
  process.env.NEXT_PUBLIC_APP_DESCRIPTION,
  'Privacy-first AI meeting assistant'
);
export const APP_COMPANY = getString(process.env.NEXT_PUBLIC_APP_COMPANY, 'Your Team');
export const APP_REPOSITORY_URL = getString(
  process.env.NEXT_PUBLIC_APP_REPOSITORY_URL,
  'https://github.com/Dennyanne/meetily_denny'
);
export const APP_SUPPORT_URL = getString(process.env.NEXT_PUBLIC_APP_SUPPORT_URL, APP_REPOSITORY_URL);
export const APP_PRIVACY_POLICY_URL = getString(
  process.env.NEXT_PUBLIC_APP_PRIVACY_POLICY_URL,
  `${APP_REPOSITORY_URL}/blob/main/PRIVACY_POLICY.md`
);
export const APP_BLUETOOTH_NOTICE_URL = getString(
  process.env.NEXT_PUBLIC_APP_BLUETOOTH_NOTICE_URL,
  `${APP_REPOSITORY_URL}/blob/main/BLUETOOTH_PLAYBACK_NOTICE.md`
);
export const APP_ENABLE_ANALYTICS = getBoolean(process.env.NEXT_PUBLIC_ENABLE_ANALYTICS, false);
export const APP_ENABLE_UPDATES = getBoolean(process.env.NEXT_PUBLIC_ENABLE_UPDATES, false);
