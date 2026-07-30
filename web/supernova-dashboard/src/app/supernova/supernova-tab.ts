import { CommonModule } from '@angular/common';
import { Component, inject, signal, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { JobRecord, SupernovaApi } from '../supernova-api';

@Component({
  selector: 'app-supernova-tab',
  imports: [CommonModule, FormsModule],
  templateUrl: './supernova-tab.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './supernova-tab.css',
})
export class SupernovaTab {
  private readonly api = inject(SupernovaApi);

  protected readonly collections = signal<string[]>([]);
  protected readonly jobs = signal<JobRecord[]>([]);
  protected readonly selectedJobId = signal('');
  protected readonly selectedJob = signal<JobRecord | null>(null);
  protected readonly loadConfigYaml = signal('');
  protected readonly stormConfigYaml = signal('');
  protected readonly error = signal('');

  protected refreshCollections(): void {
    this.api.getCollections().subscribe({
      next: (res) => {
        this.collections.set(res.collections);
        this.error.set('');
      },
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to load collections'),
    });
  }

  protected submitLoad(): void {
    this.api.submitLoadRun(this.loadConfigYaml()).subscribe({
      next: () => this.refreshJobs(),
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to submit load job'),
    });
  }

  protected submitStorm(): void {
    this.api.submitStormRun(this.stormConfigYaml()).subscribe({
      next: () => this.refreshJobs(),
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to submit storm job'),
    });
  }

  protected refreshJobs(): void {
    this.api.listJobs().subscribe({
      next: (jobs) => {
        this.jobs.set(jobs);
        this.error.set('');
      },
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to load jobs'),
    });
  }

  protected openJob(): void {
    const id = this.selectedJobId().trim();
    if (!id) {
      return;
    }
    this.api.getJob(id).subscribe({
      next: (job) => {
        this.selectedJob.set(job);
        this.error.set('');
      },
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to load job'),
    });
  }

  protected cancelJob(id: string): void {
    this.api.cancelJob(id).subscribe({
      next: () => this.refreshJobs(),
      error: (err) => this.error.set(err?.error?.error ?? 'Failed to cancel job'),
    });
  }
}
