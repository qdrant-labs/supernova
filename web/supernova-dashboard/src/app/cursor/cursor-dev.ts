import { Component, inject, PLATFORM_ID, signal, ChangeDetectionStrategy } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CursorDevService } from './cursor-dev.service';

@Component({
  selector: 'app-cursor-dev',
  imports: [FormsModule],
  templateUrl: './cursor-dev.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './cursor-dev.css',
})
export class CursorDev {
  private readonly cursorDevService = inject(CursorDevService);
  private readonly platformId = inject(PLATFORM_ID);

  protected readonly prompt = signal('');
  protected readonly response = signal('');
  protected readonly status = signal('Checking Cursor agent...');
  protected readonly busy = signal(false);

  constructor() {
    if (isPlatformBrowser(this.platformId)) {
      this.refreshStatus();
      return;
    }

    this.status.set('Open in the browser and run `npm run dev:cursor-api`.');
  }

  refreshStatus(): void {
    this.cursorDevService.getStatus().subscribe({
      next: (result) => {
        if (result.ready && result.agentId) {
          this.status.set(`Agent ready: ${result.agentId}`);
          return;
        }

        this.status.set(result.error ?? 'Cursor agent is not ready.');
      },
      error: () => {
        this.status.set('Cursor API unavailable. Run `npm run dev:cursor-api` with CURSOR_API_KEY set.');
      },
    });
  }

  sendPrompt(): void {
    const message = this.prompt().trim();
    if (!message || this.busy()) {
      return;
    }

    this.busy.set(true);
    this.response.set('');

    this.cursorDevService.chat(message).subscribe({
      next: (result) => {
        this.response.set(result.text);
        this.status.set(`Run ${result.runId} finished with status ${result.status}`);
        this.busy.set(false);
      },
      error: (error) => {
        const messageText =
          typeof error?.error?.error === 'string'
            ? error.error.error
            : 'Failed to send prompt to Cursor agent.';
        this.status.set(messageText);
        this.busy.set(false);
      },
    });
  }
}
