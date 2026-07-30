import { Component, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-user-profile',
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <div class="card">
      <div class="card-body">
        <h3 class="h5">User Profile</h3>
        <p class="mb-0">Profile tab preserved from the source dashboard.</p>
      </div>
    </div>
  `,
})
export class UserProfile {}
