# Kamidso Codex Skills Pack

چهار Skill اصلی استخراج‌شده از پرامپت‌های تکراری پروژه Kamidso Lite، با ساختار استاندارد `SKILL.md` و دستورهای اجرایی انگلیسی برای Codex.

## شامل

- `$kamidso-tdd`: **توسعه Skill موجود** با حفظ متن فعلی، دروازه‌های تأیید RED/GREEN، محدودیت Scope و الزام گزارش‌دهی.
- `$kamidso-test-runner`: اجرای تست آفلاین، Docker، مقایسه Baseline و گزارش تست‌ها.
- `$kamidso-evidence-grounding`: Grounding، منشأ Evidence، قرارداد سه Recovery و ارتباط Python/PNG.
- `$kamidso-live-qualification`: فقط تست Live صریحاً مجاز، Preflight، یک Node و یک اجرای کنترل‌شده و Diagnostic امن.

## نصب روی Windows در پوشه پروژه

فایل ZIP را Extract کنید. با Python نصب کنید:

```powershell
python .\kamidso-codex-skills\scripts\validate_skills.py
python .\kamidso-codex-skills\install_kamidso_skills.py --repo "D:\programming\freeLanser\VIRA_ASRE\enterprise-data-analyst-agent" --dry-run
python .\kamidso-codex-skills\install_kamidso_skills.py --repo "D:\programming\freeLanser\VIRA_ASRE\enterprise-data-analyst-agent"
```

اگر بسته را در مسیر دیگری Extract کرده‌اید، مسیر اسکریپت را مطابق همان محل تغییر دهید. اسکریپت آفلاین است و به API یا GitHub وصل نمی‌شود؛ `AGENTS.md` پروژه را بررسی می‌کند و Skillها را در `.agents/skills/` قرار می‌دهد.

### حفظ Skill موجود

- اگر `kamidso-tdd/SKILL.md` از قبل وجود داشته باشد، installer **آن را جایگزین نمی‌کند**؛ فقط بخش مشخص‌شده با `KAMIDSO_TDD_PACK_EXTENSION` را اضافه می‌کند و پیش از آن نسخه `.bak` می‌سازد.
- اجرای دوباره installer، extension را دوباره اضافه نمی‌کند.
- اگر سه Skill جدید از قبل وجود داشته باشند، installer به‌طور پیش‌فرض آن‌ها را **overwrite نمی‌کند**.
- برای نصب صرفاً سه Skill جدید و بدون تغییر TDD از `--skip-tdd-extension` استفاده کنید.
- قبل از Commit، `git diff` را بررسی کنید. پس از نصب، یک جلسه جدید Codex باز کنید.

## مثال پرامپت کوتاه

```text
Use $kamidso-tdd, $kamidso-test-runner,
$kamidso-evidence-grounding, and $kamidso-live-qualification.

PHASE: 4-C-C.5-R (approved controlled live qualification)
Approved exact node:
live_tests/test_agent_end_to_end_live.py::test_live_python_png_and_conversation_reuse

Scope: verify real Qwen Python execution, PNG,
strict grounding, restart and exact evidence reuse.
Use the approved local budgets and model unchanged.
Run this exact live test once only, capture redacted diagnostics,
report results, and STOP for human review.
```

این **صرفاً مثال یک دستور** است، نه مجوز کنونی برای اجرای Live. بدون تأیید صریح جدید نباید درخواست billable فرستاده شود.

## نکات امنیتی و قراردادی

- `AGENTS.md` پروژه اولویت دارد. Skillها Scope یا اختیار جدیدی صادر نمی‌کنند.
- مدل، Budgetها، نام دقیق Node و تعداد تست‌ها داده‌های هر فاز هستند و نباید از گزارش‌های قدیمی به‌عنوان مقدار ثابت استفاده شوند.
- بسته هیچ Dataset، توکن API یا فایل `.env` ندارد و هیچ Live Test اجرا نمی‌کند.
- فایل‌های `.agents/skills/*/references/*.md` فقط در صورت نیاز به جزئیات مرتبط خوانده می‌شوند.
