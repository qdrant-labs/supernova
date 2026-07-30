import { Component, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-perf',
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <div class="card">
      <div class="card-body">
        <h2 class="h4">Performance</h2>
        <p class="mb-0">Performance tab preserved from the source dashboard.</p>
      </div>
    </div>
  `,
})
export class Perf {}
