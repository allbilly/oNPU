"""Browser QA against a real running server with a ready model."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", default="test-results")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto(args.url)
        page.locator("#connection-text").filter(has_text="Server online").wait_for()
        for route in ("overview", "models", "chat", "benchmarks", "settings"):
            page.locator(f'nav [data-view="{route}"]').click()
            page.locator(f"#view-{route}").wait_for(state="visible")
            assert page.locator("h1:visible").inner_text().lower() == route
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), route
        page.locator("#setting-context").fill("512")
        page.locator("#setting-max-tokens").fill("64")
        page.locator("#setting-idle").fill("0")
        page.locator("#settings-form button[type=submit]").click()
        page.locator("#settings-saved").filter(has_text="Saved").wait_for()
        page.reload()
        page.locator("#view-settings").wait_for(state="visible")
        assert page.locator("#setting-context").input_value() == "512"
        page.locator('[data-view="chat"]').click()
        page.locator("#new-chat").click()
        page.locator("#chat-options-toggle").click()
        page.locator("#chat-max-tokens").fill("8")
        page.locator("#chat-temperature").fill("0")
        page.locator("#chat-input").fill("Reply with exactly the word hello.")
        page.locator("#send-message").click()
        page.locator(".message.assistant").wait_for()
        page.locator("#stop-generation").wait_for(state="hidden", timeout=180000)
        reply = page.locator(".message.assistant .message-content").inner_text()
        assert reply.strip(), reply
        assert page.locator(".message-error").count() == 0
        print("browser reply:", repr(reply), flush=True)
        page.screenshot(path=str(out / "chat.png"), full_page=True, animations="disabled")
        page.reload()
        page.locator(".message.assistant").wait_for()
        assert page.locator(".message.assistant .message-content").inner_text() == reply
        # Stop is tested through the real button while a long reply is pending.
        page.locator("#chat-input").fill("Count every number from 1 to 100 and explain each.")
        page.locator("#send-message").click()
        page.locator("#stop-generation").wait_for(state="visible")
        page.locator("#stop-generation").click()
        page.locator(".message-error").filter(has_text="Stopped").wait_for(timeout=45000)
        for _ in range(90):
            if page.request.get(args.url + "/api/status").json()["runtime"]["state"] == "ready":
                break
            time.sleep(0.5)
        else:
            raise AssertionError("Stop did not release the worker")
        # Exercise imports and the manager's actual load/unload forms.
        runtime = page.request.get(args.url + "/api/status").json()["runtime"]
        model_id = runtime["model_id"]
        model = next(item for item in page.request.get(args.url + "/api/models").json()["models"] if item["id"] == model_id)
        page.locator('[data-view="models"]').click()
        page.locator("#view-models").wait_for(state="visible")
        page.locator('#model-library [data-action="unload"]').click()
        page.locator(f'#model-library [data-action="load"][data-id="{model_id}"]').wait_for()
        assert page.request.get(args.url + "/api/status").json()["runtime"]["rss_bytes"] == 0
        page.locator("#import-open").click()
        page.locator("#import-path").fill(model["path"])
        page.locator('#import-form button[type="submit"], #import-form button.button.primary').click()
        page.locator("#import-dialog").wait_for(state="hidden")
        page.locator(f'#model-library [data-action="load"][data-id="{model_id}"]').click()
        page.locator("#load-profile").select_option("CPU")
        page.locator("#load-context").fill("512")
        page.locator("#load-form button.button.primary").click()
        page.locator("#model-dialog").wait_for(state="hidden")
        for _ in range(240):
            state = page.request.get(args.url + "/api/status").json()["runtime"]
            assert state["state"] != "error", state
            if state["state"] == "ready":
                assert state["actual_device"] == "CPU"
                break
            time.sleep(0.5)
        else:
            raise AssertionError("UI model load did not become ready")
        print("browser Stop, import, unload and load passed", flush=True)
        page.locator('[data-view="overview"]').click()
        page.screenshot(path=str(out / "overview.png"), full_page=True, animations="disabled")
        page.locator("#theme-toggle").click()
        assert page.locator("html").get_attribute("data-theme") == "dark"
        page.screenshot(path=str(out / "dark.png"), full_page=True, animations="disabled")
        page.locator("#theme-toggle").click()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.locator(".sidebar-bottom").is_visible()
        for route in ("overview", "models", "chat", "benchmarks", "settings"):
            page.locator(f'nav [data-view="{route}"]').click()
            page.locator(f"#view-{route}").wait_for(state="visible")
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), route
        page.locator('[data-view="overview"]').click()
        page.screenshot(path=str(out / "mobile.png"), full_page=True, animations="disabled")
        assert not errors, errors
        browser.close()
        print("PASS: real browser chat, Stop, model lifecycle, history, saved settings, themes, desktop/mobile navigation; no console errors.", flush=True)


if __name__ == "__main__":
    main()
